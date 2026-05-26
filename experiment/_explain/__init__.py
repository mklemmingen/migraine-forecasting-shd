"""Shared explainability module for the Addition-2 insight pass.

``emit_insights`` is the single entry point the evaluate templates call
when ``EMIT_INSIGHTS`` is set. It is off by default, so a normal sweep
pays no cost. It computes the per-leaf attribution / effect / interaction /
embedding artefacts and writes them into the leaf's ``insights/`` folder,
plus a machine-parseable ``explain_<ts>.txt`` summary.

Architecture-matched estimators (docs Section 3.1):
  - xgboost    : KernelSHAP over the calibrated-probability closure
  - tabpfn     : TabPFN-native SHAP + ShapIQ interactions + embedding
  - autotabpfn : permutation importance over the public predict_proba

The signature takes a ``predict_fn`` closure and a background frame rather
than an ``X_train`` argument: the evaluate templates never load the train
set, so the SHAP background is the val/cal set already in scope.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Number of top features that receive an ALE effect plot.
_ALE_TOP_K = 5
# Beeswarm omitted above this row count to stay readable (docs Section 3.1).
_BEESWARM_MAX_ROWS = 1500


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _free_gpu() -> None:
    """Release cached GPU memory between heavy attribution stages.

    The TabPFN paths (native SHAP, ShapIQ coalition refits, embedding) each
    allocate sizeable HIP buffers; freeing the cache between them keeps the
    AMD ROCm runtime from accumulating allocations across stages and
    crashing the process mid-pass (the failure mode is a native abort that
    no Python try/except can intercept)."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def emit_insights(
    predict_fn,
    X_background,
    X_explain,
    y_explain,
    leaf_dir,
    arch_family,
    raw_model=None,
    title: str | None = None,
    y_background=None,
) -> Path:
    """Compute and write the per-leaf insight artefacts. Returns the
    ``insights/`` directory path.

    Failures inside any single artefact are isolated so one broken figure
    cannot abort the others or corrupt the metrics already written on the
    clean evaluate run.
    """
    from _explain import shap_runners
    from _explain import ale as ale_mod
    from _explain import _plots

    insights_dir = Path(leaf_dir) / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    title = title or Path(leaf_dir).name
    if not isinstance(X_background, pd.DataFrame):
        X_background = pd.DataFrame(X_background)
    if not isinstance(X_explain, pd.DataFrame):
        X_explain = pd.DataFrame(X_explain)

    print(f"[insights] {title} | arch_family={arch_family} | "
          f"bg={len(X_background)} explain={len(X_explain)}", flush=True)

    # --- Attribution (SHAP or permutation importance) -----------------------
    feature_names, matrix, attr_meta = shap_runners.compute_attributions(
        predict_fn, X_background, X_explain, y_explain, arch_family, raw_model=raw_model,
    )
    ranking = shap_runners.mean_abs_ranking(feature_names, matrix)
    is_perm = attr_meta.get("is_permutation", False)
    metric_label = "mean |permutation importance|" if is_perm else "mean |SHAP|"

    # Persist the complete ranking (the text summary keeps only the top-10);
    # the Park-OR and prodromal-contamination checks need every feature.
    from _explain.summarise import write_ranking_csv
    _safe(write_ranking_csv, insights_dir, ts, ranking, metric_label)

    _safe(_plots.plot_attribution_bar, ranking, metric_label,
          f"{title} - feature attribution", insights_dir / f"shap_bar_{ts}.png")

    # Beeswarm only for true per-row SHAP matrices (not the permutation row)
    # and only when the row count stays readable.
    if not is_perm and matrix.shape[0] > 1 and matrix.shape[0] <= _BEESWARM_MAX_ROWS:
        # Feature values aligned with the SHAP-matrix columns, for value-coloured
        # beeswarm points (missing feature -> NaN -> mid colour).
        feat_vals = X_explain.reindex(columns=feature_names).to_numpy(dtype=float)
        # Freeze the matrix + values so the paper beeswarm (fig_h docs script) can
        # re-render without recomputing SHAP; the PNG alone is not reusable data.
        _safe(np.savez_compressed, str(insights_dir / f"shap_matrix_{ts}.npz"),
              feature_names=np.array(feature_names, dtype=object),
              shap=matrix, values=feat_vals)
        _safe(_plots.plot_beeswarm, feature_names, matrix, feat_vals,
              f"{title} - per-row SHAP", insights_dir / f"shap_beeswarm_{ts}.png")

    # --- ALE on the top-K attributed features -------------------------------
    ale_slopes: list[tuple[str, float]] = []
    top_feats = [f for f, _ in ranking[:_ALE_TOP_K]]
    for feat in top_feats:
        try:
            curve = ale_mod.first_order_ale(predict_fn, X_explain, feat)
            grid = np.quantile(X_explain[feat].to_numpy(dtype=float),
                               np.linspace(0.05, 0.95, 10))
            grid = np.unique(grid)
            pdp_vals = ale_mod.partial_dependence(predict_fn, X_explain, feat, grid)
            _safe(_plots.plot_ale, curve, grid, pdp_vals, feat,
                  f"{title} - ALE: {feat}", insights_dir / f"ale_{feat}_{ts}.png")
            ale_slopes.append((feat, ale_mod.ale_slope(curve)))
        except Exception as exc:  # pragma: no cover - per-feature isolation
            print(f"[insights] ALE failed for {feat}: {type(exc).__name__}: {exc}", flush=True)

    # Write the attribution + ALE summary now, before the GPU-heavy optional
    # TabPFN stages. A native ROCm crash inside ShapIQ or the embedding
    # projection kills the process without a catchable exception; persisting
    # the summary here guarantees the SHAP/ALE result survives such a crash.
    # The full summary (with interactions/embedding) overwrites it on success.
    from _explain.summarise import write_summary
    write_summary(insights_dir, ts, title, arch_family, ranking, attr_meta,
                  ale_slopes, None, None, None)

    # --- ShapIQ pairwise interactions (TabPFN single-fit headlines only) ----
    interaction_pairs = None
    interaction_meta = None
    if arch_family == "tabpfn" and raw_model is not None:
        _free_gpu()
        try:
            from _explain import shapiq_runner
            # ShapIQ's TabPFN explainer conditions on the background set, so
            # the labels must match X_background (the val/cal set), not the
            # test labels. y_background falls back to y_explain only when the
            # template did not plumb it (older leaves).
            shapiq_labels = y_background if y_background is not None else y_explain
            result = shapiq_runner.compute_pairwise_interactions(
                raw_model, X_background, shapiq_labels, X_explain,
                predict_fn=predict_fn,
            )
            interaction_pairs = shapiq_runner.top_pair_labels(result, n=10)
            interaction_meta = result["meta"]
            _safe(_plots.plot_interactions, result,
                  f"{title} - pairwise interactions", insights_dir / f"shapiq_interactions_{ts}.png")
        except Exception as exc:
            print(f"[insights] ShapIQ failed: {type(exc).__name__}: {exc}", flush=True)

    # --- Embedding projection (TabPFN single-fit only) ----------------------
    embedding_meta = None
    if arch_family == "tabpfn" and raw_model is not None:
        _free_gpu()
        try:
            from _explain import embedding as emb_mod
            proba = predict_fn(X_explain)
            projection = emb_mod.compute_embedding_projection(raw_model, X_explain, proba)
            if projection is not None:
                embedding_meta = projection["meta"]
                _safe(_plots.plot_embedding, projection,
                      f"{title} - attention embedding", insights_dir / f"embedding_{ts}.png")
        except Exception as exc:
            print(f"[insights] embedding failed: {type(exc).__name__}: {exc}", flush=True)

    # --- Per-leaf text summary (full: overwrites the early SHAP/ALE-only one)
    summary_path = write_summary(
        insights_dir, ts, title, arch_family, ranking, attr_meta,
        ale_slopes, interaction_pairs, interaction_meta, embedding_meta,
    )
    print(f"[insights] wrote {summary_path}", flush=True)
    return insights_dir


def _safe(fn, *args, **kwargs) -> None:
    """Run a plotting helper, isolating any failure to that one figure."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - per-figure isolation
        print(f"[insights] {getattr(fn, '__name__', fn)} failed: "
              f"{type(exc).__name__}: {exc}", flush=True)
