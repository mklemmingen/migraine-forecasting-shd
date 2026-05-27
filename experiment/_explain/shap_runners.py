"""Per-architecture SHAP / permutation-importance dispatch.

Returns a uniform ``(feature_names, shap_matrix)`` tuple so the downstream
plotting and summary code is architecture-agnostic. ``shap_matrix`` has
shape ``(n_explain, n_features)`` and carries signed attributions on the
quantity the benchmark scores: the positive-class probability.

Estimator choice per architecture family (rationale in
docs/addition2_explainability.md Section 3.1):

- ``xgboost`` (Addition 0): model-agnostic KernelSHAP over the leaf's
  ``calibrated_proba(bundle, .)`` closure. The Addition-0 architectures
  are calibrated stacking/blending pipelines, not single trees, so
  TreeSHAP cannot represent the meta-learner, the blend, or the
  calibrators. KernelSHAP on the calibrated-probability closure explains
  exactly the probability the benchmark compares across architectures.
- ``tabpfn`` (Addition 1, single-fit): the TabPFN-native explainer in
  ``tabpfn_extensions.interpretability.shap`` builds a SHAP explainer over
  the in-context forward pass rather than a generic KernelSHAP wrapper.
- ``autotabpfn`` (Addition 1, post-hoc ensemble): permutation importance
  via the handle's public ``predict_proba``. The AutoGluon post-hoc
  ensemble has no single differentiable forward pass, so permutation
  importance over the public interface is the model-matched alternative;
  it is reported as permutation importance, never compared 1:1 with
  Shapley values.

Sampling-based estimators are seeded; the background-sample count is
reported by the caller via the returned ``meta`` dict.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Fixed seeds so the sampling-based estimators are reproducible across
# re-runs of the insight pass.
SHAP_SEED = 42
# KernelSHAP cost scales with the background size; the val/cal sets are
# 439 rows on the canonical chrono cells, so a k-means summary keeps the
# estimator tractable while covering the background distribution.
_BACKGROUND_K = 50
# Cap the explained rows so a single leaf's KernelSHAP stays inside the
# per-leaf compute budget; the chrono test sets are ~136 rows, well under
# this, so no subsampling happens on the canonical cells.
_MAX_EXPLAIN = 400
# The TabPFN-native explainer issues one in-context forward pass per
# perturbation, which on the high-dimensional feature sets (52 features ->
# shap's PermutationExplainer) costs ~10-15s per explained row on GPU.
# The mean-absolute-SHAP ranking is an average over rows, so a smaller
# explained sample yields a stable ranking at a tractable cost; the cap is
# applied only to the TabPFN path (KernelSHAP over the cheap XGBoost closure
# stays at the larger cap). The sampled rows are reported in the metadata.
_MAX_EXPLAIN_TABPFN = 40


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _subsample(X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=n, replace=False))
    return X.iloc[idx]


def _summarise_background(X_background: pd.DataFrame, k: int):
    """Compress the background frame to at most ``k`` representative rows.

    KernelSHAP's cost scales with the background size, so a k-means summary
    keeps it tractable. ``shap.kmeans`` requires the cluster count not exceed
    the number of distinct rows; the low-cardinality binary feature sets
    (e.g. the 6-feature park cell) collapse to far fewer distinct rows than
    the row count, so the cluster count is capped at the distinct-row count.
    When even that is degenerate the unique rows are used directly so the
    summary never silently drops the background distribution.
    """
    import shap

    distinct = np.unique(X_background.values, axis=0)
    k_eff = int(max(1, min(k, len(distinct))))
    if k_eff >= len(distinct):
        # Fewer distinct rows than the requested summary size: use the
        # distinct rows directly (a faithful, exact background).
        return distinct, k_eff, "distinct-rows"
    return shap.kmeans(X_background.values, k_eff), k_eff, "k-means"


def _kernel_shap_calibrated(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> tuple[list[str], np.ndarray, dict]:
    """Model-agnostic KernelSHAP over a probability closure.

    ``predict_fn`` maps a feature frame to the positive-class probability
    vector. A k-means (or distinct-row) summary of the background keeps
    KernelSHAP tractable; the background is the val/cal set already in
    evaluate's scope, never the train set.
    """
    import shap

    np.random.seed(SHAP_SEED)
    bg, k_eff, summary_kind = _summarise_background(X_background, _BACKGROUND_K)

    def _f(arr: np.ndarray) -> np.ndarray:
        return predict_fn(pd.DataFrame(arr, columns=X_explain.columns))

    explainer = shap.KernelExplainer(_f, bg)
    values = explainer.shap_values(X_explain.values, nsamples="auto", silent=True)
    matrix = np.asarray(values)
    # KernelExplainer on a single-output closure returns a 2D matrix; some
    # shap versions wrap it in a length-1 list. Normalise to (n, features).
    if isinstance(values, list):
        matrix = np.asarray(values[-1])
    if matrix.ndim == 3:
        matrix = matrix[:, :, -1]
    meta = {
        "estimator": "KernelSHAP (model-agnostic, over calibrated probability)",
        "background_n": int(len(X_background)),
        "background_summary_k": k_eff,
        "background_summary": summary_kind,
        "explained_n": int(len(X_explain)),
        "seed": SHAP_SEED,
    }
    return list(X_explain.columns), matrix, meta


def _tabpfn_native_shap(
    raw_model,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
) -> tuple[list[str], np.ndarray, dict]:
    """TabPFN-native SHAP via the interpretability extension.

    ``get_shap_values`` selects the TabPFN explainer when the estimator is
    a TabPFN model and queries its in-context forward pass directly. The
    returned object exposes signed per-class values; the positive-class
    slice is extracted to match the benchmark's scored quantity.
    """
    import shap  # noqa: F401  (the extension lazily imports shap internally)
    from tabpfn_extensions.interpretability.shap import get_shap_values

    np.random.seed(SHAP_SEED)
    explanation = get_shap_values(
        raw_model,
        X_explain,
        attribute_names=list(X_explain.columns),
    )
    values = getattr(explanation, "values", explanation)
    matrix = np.asarray(values)
    # SHAP returns (n, features) for a scalar output or (n, features, classes)
    # for a multi-output explainer; take the positive-class slice when 3D.
    if matrix.ndim == 3:
        matrix = matrix[:, :, -1]
    meta = {
        "estimator": "TabPFN-native SHAP (tabpfn_extensions.interpretability)",
        "background_n": int(len(X_background)),
        "explained_n": int(len(X_explain)),
        "seed": SHAP_SEED,
    }
    return list(X_explain.columns), matrix, meta


def _permutation_importance(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X_explain: pd.DataFrame,
    y_explain,
    n_repeats: int = 10,
) -> tuple[list[str], np.ndarray, dict]:
    """Permutation importance over the public probability closure.

    Used for the AutoTabPFN post-hoc ensemble, which has no single
    forward pass. Importance is the mean AUROC drop when a feature column
    is shuffled. The returned matrix is a single-row, non-negative
    importance vector (broadcast to the uniform 2D shape) clearly labelled
    as permutation importance, not Shapley attribution.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_explain)
    rng = np.random.default_rng(SHAP_SEED)
    base = roc_auc_score(y, predict_fn(X_explain)) if len(np.unique(y)) > 1 else 0.5
    cols = list(X_explain.columns)
    importances = np.zeros(len(cols))
    for j, col in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            shuffled = X_explain.copy()
            shuffled[col] = rng.permutation(shuffled[col].values)
            score = roc_auc_score(y, predict_fn(shuffled)) if len(np.unique(y)) > 1 else 0.5
            drops.append(base - score)
        importances[j] = float(np.mean(drops))
    meta = {
        "estimator": "permutation importance (AUROC drop over public predict_proba)",
        "explained_n": int(len(X_explain)),
        "n_repeats": n_repeats,
        "base_auroc": float(base),
        "seed": SHAP_SEED,
        "is_permutation": True,
    }
    return cols, importances.reshape(1, -1), meta


def compute_attributions(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X_background,
    X_explain,
    y_explain,
    arch_family: str,
    raw_model=None,
) -> tuple[list[str], np.ndarray, dict, pd.DataFrame]:
    """Dispatch to the architecture-matched attribution estimator.

    Returns ``(feature_names, matrix, meta, X_explain_used)``.

    ``matrix`` is ``(n_explained, n_features)`` of signed SHAP values for the
    SHAP paths, or a single-row non-negative permutation-importance vector for
    the AutoTabPFN path (``meta['is_permutation']`` flags this).

    ``X_explain_used`` is the row subset of the input ``X_explain`` that was
    actually passed to the attribution estimator after subsampling. Its i-th
    row corresponds to ``matrix[i]``, so callers that need per-row feature
    values (e.g. beeswarm colour encoding) must index from ``X_explain_used``,
    not from the original ``X_explain``.
    """
    X_background = _as_frame(X_background)
    # The TabPFN path pays a forward pass per perturbation, so it uses a
    # tighter explained-row cap than the cheap XGBoost-closure KernelSHAP.
    cap = _MAX_EXPLAIN_TABPFN if arch_family == "tabpfn" else _MAX_EXPLAIN
    X_explain = _subsample(_as_frame(X_explain), cap, SHAP_SEED)

    if arch_family == "xgboost":
        feature_names, matrix, meta = _kernel_shap_calibrated(
            predict_fn, X_background, X_explain)
        return feature_names, matrix, meta, X_explain
    if arch_family == "tabpfn":
        feature_names, matrix, meta = _tabpfn_native_shap(
            raw_model, X_background, X_explain)
        return feature_names, matrix, meta, X_explain
    if arch_family == "autotabpfn":
        y = y_explain.loc[X_explain.index] if hasattr(y_explain, "loc") else y_explain
        feature_names, matrix, meta = _permutation_importance(predict_fn, X_explain, y)
        return feature_names, matrix, meta, X_explain
    raise ValueError(f"unknown arch_family for attribution: {arch_family!r}")


def mean_abs_ranking(feature_names: list[str], matrix: np.ndarray) -> list[tuple[str, float]]:
    """Rank features by mean absolute attribution, descending."""
    mean_abs = np.abs(matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [(feature_names[i], float(mean_abs[i])) for i in order]
