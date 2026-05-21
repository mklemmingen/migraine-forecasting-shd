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

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
sys.path.insert(0, str(_REPO_ROOT / "data" / "pipeline" / "analytics"))
try:
    from style import PALETTE, POS_COLOR, NEG_COLOR, apply_theme  # noqa: E402
except Exception:  # pragma: no cover - style import is best-effort
    PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7"]
    POS_COLOR, NEG_COLOR = "#D55E00", "#56B4E9"

    def apply_theme() -> None:
        pass


def plot_attribution_bar(ranking, metric_label, title, out_path, top_n=15):
    """Horizontal bar of mean absolute attribution for the top features."""
    apply_theme()
    items = ranking[:top_n][::-1]
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(names) + 1)))
    ax.barh(range(len(names)), vals, color=PALETTE[1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(metric_label)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_beeswarm(feature_names, matrix, title, out_path, top_n=12):
    """Per-row signed-attribution strip plot for the top features.

    Omitted by the caller when n > 1500 rows to stay readable. Points are
    coloured by feature value (high = positive colour, low = negative).
    """
    apply_theme()
    mean_abs = np.abs(matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(order) + 1)))
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        vals = matrix[:, j]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(vals, np.full(len(vals), row) + jitter, s=8, alpha=0.5,
                   color=PALETTE[0], edgecolors="none")
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[j] for j in order], fontsize=8)
    ax.set_xlabel("SHAP value (impact on positive-class probability)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
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
