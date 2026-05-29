"""Matplotlib figure helpers for the per-leaf insight pass.

Reuses the Okabe-Ito colour-blind-safe palette from
``data/pipeline/analytics/style.py`` for visual consistency with the rest
of the study. All figures are written to the leaf's ``insights/`` folder
with a timestamp so re-runs do not overwrite prior outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


def _figure_style():
    """Lazy-import the repo's single figure-style module (experiment/_style.py)
    for the paper-grade attribution/beeswarm figures, so they share the brand
    palette, fonts, and the vector-PDF + 300 dpi PNG save() contract. Imported
    by file path so the insight pass need not have experiment/ on sys.path."""
    exp = Path(__file__).resolve().parents[1]   # experiment/
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import _style
    return _style

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
sys.path.insert(0, str(_REPO_ROOT / "data" / "pipeline" / "analytics"))
try:
    from style import PALETTE, POS_COLOR, NEG_COLOR, apply_theme  # noqa: E402
except Exception:  # pragma: no cover - style import is best-effort
    PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7"]
    POS_COLOR, NEG_COLOR = "#D55E00", "#56B4E9"

    def apply_theme() -> None:
        pass


def plot_attribution_bar(ranking, metric_label, title, out_path, top_n=15, ci=None):
    """Horizontal bar of mean absolute attribution for the top features.

    Brand-styled (figure _style): blue bars, double-column width, vector PDF +
    300 dpi PNG via save(). ``out_path`` may be a .png stem; both siblings write.

    When ``ci`` is provided it must map ``feature_name -> (lo, hi)`` for the
    per-bar bootstrap envelope; whiskers render at xerr asymmetric from the bar
    centre, and CI brackets append to each y-tick label.
    """
    S = _figure_style()
    S.apply()
    items = ranking[:top_n][::-1]
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=S.figsize("double", max(4.0, 0.4 * len(names) + 1)))
    ax.barh(range(len(names)), vals, color=S.OI["blue"])
    if ci is not None:
        lo_arr = np.array([max(0.0, vals[i] - ci.get(names[i], (vals[i], vals[i]))[0])
                           for i in range(len(names))])
        hi_arr = np.array([max(0.0, ci.get(names[i], (vals[i], vals[i]))[1] - vals[i])
                           for i in range(len(names))])
        ax.errorbar(vals, range(len(names)), xerr=[lo_arr, hi_arr],
                    fmt="none", ecolor=S.REF_COLOR, elinewidth=0.9, capsize=2.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(metric_label)
    ax.set_title(title, fontsize=10)
    S.cc_by_footer(fig)
    S.save(fig, out_path)
    plt.close(fig)


def attribution_row_bootstrap_ci(shap_matrix, feature_names, n_boot=500, seed=42, alpha=0.05):
    """Row-bootstrap 95% CI on mean(|SHAP|) per feature.

    Resamples test rows (patient-days) with replacement n_boot times and
    recomputes the per-feature mean of |SHAP|. Returns a dict mapping the
    feature name to (lo, hi). This is a row-stability CI on a fixed model,
    not a model-stability CI (which would require N model re-fits).
    """
    arr = np.abs(np.asarray(shap_matrix, dtype=float))
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, arr.shape[1]), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = arr[idx].mean(axis=0)
    lo = np.percentile(boot, 100.0 * alpha / 2.0, axis=0)
    hi = np.percentile(boot, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return {str(feature_names[j]): (float(lo[j]), float(hi[j]))
            for j in range(arr.shape[1])}


def plot_beeswarm(feature_names, matrix, feature_values, title, out_path, top_n=12):
    """Per-row SHAP beeswarm for the top features, points coloured by the
    feature's own value (low = blue, high = vermillion), the standard SHAP
    encoding that shows the direction of each feature's effect.

    Omitted by the caller when n > 1500 rows to stay readable. ``feature_values``
    is the row x feature value matrix aligned with ``matrix``'s columns; pass an
    all-NaN array of the same shape if values are unavailable (points then render
    in the mid colour). Brand-styled (figure _style); vector PDF + 300 dpi PNG.
    """
    S = _figure_style()
    S.apply()
    mean_abs = np.abs(matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=S.figsize("double", max(4.0, 0.45 * len(order) + 1)))
    # Low -> high feature value: blue -> grey -> vermillion (the brand diverging
    # pair; CVD-safe, unlike SHAP's default red/blue against a red background).
    cmap = LinearSegmentedColormap.from_list(
        "shd_lowhigh", [S.OI["blue"], S.FAINT, S.OI["vermillion"]])
    rng = np.random.default_rng(0)
    sc = None
    for row, j in enumerate(order):
        sv = matrix[:, j]
        fv = feature_values[:, j].astype(float)
        lo, hi = np.nanpercentile(fv, [2, 98]) if np.isfinite(fv).any() else (0.0, 1.0)
        norm = np.clip((fv - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        jitter = rng.uniform(-0.18, 0.18, size=len(sv))
        sc = ax.scatter(sv, np.full(len(sv), row) + jitter, c=norm, cmap=cmap,
                        vmin=0.0, vmax=1.0, s=8, alpha=0.6, edgecolors="none")
    ax.axvline(0.0, color=S.REF_COLOR, lw=S.REF_LW)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[j] for j in order], fontsize=8)
    ax.set_xlabel("SHAP value (impact on positive-class probability)")
    ax.set_title(title, fontsize=10)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["low", "high"])
        cbar.set_label("feature value", fontsize=8)
    S.cc_by_footer(fig)
    S.save(fig, out_path)
    plt.close(fig)


def plot_ale(curve, pdp_grid, pdp_vals, feature, title, out_path):
    """ALE curve with an optional PDP overlay for the agreement check."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(7, 5))
    centres = 0.5 * (curve["x"][:-1] + curve["x"][1:])
    ale_centres = 0.5 * (curve["ale"][:-1] + curve["ale"][1:])
    ax.plot(centres, ale_centres, marker="o", color=PALETTE[3], label="ALE (conditional)")
    if pdp_grid is not None and pdp_vals is not None:
        ax.plot(pdp_grid, pdp_vals, marker="s", linestyle="--", color=PALETTE[4],
                label="PDP (marginal, overlay)")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel("centred effect on positive-class probability")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_interactions(result, title, out_path, top_k=12):
    """Heatmap of mean absolute pairwise interaction strength (top features)."""
    apply_theme()
    names = result["feature_names"]
    mat = result["pair_matrix"]
    strength = mat.sum(axis=1)
    order = np.argsort(strength)[::-1][:top_k]
    sub = mat[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sub, cmap="magma")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="mean |k-SII order-2 interaction|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_embedding(projection, title, out_path):
    """2D embedding scatter coloured by predicted probability, with the
    mandatory over-interpretation caveat in the caption."""
    apply_theme()
    coords = projection["coords"]
    proba = projection["proba"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=proba, cmap="viridis", s=18, alpha=0.8)
    ax.set_xlabel(f"{projection['method']} dim 1")
    ax.set_ylabel(f"{projection['method']} dim 2")
    ax.set_title(title, fontsize=10)
    fig.colorbar(sc, ax=ax, label="predicted P(migraine)")
    fig.text(0.5, 0.005,
             "Qualitative aid only: inter-cluster distances in any 2D "
             "embedding must not be over-read.",
             ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
