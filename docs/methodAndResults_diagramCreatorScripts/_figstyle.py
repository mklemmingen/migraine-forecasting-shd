"""Shared matplotlib styling for the paper figures.

Consistent fonts/sizes, a fixed colour palette for targets and architectures, and
a save() that writes PNG (raster preview) + PDF (vector, for LaTeX) for every
figure. Imported by the fig_*.py diagram-creator scripts.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

TARGET = {"headache": "#1f77b4", "migraine": "#d62728"}
ARCH = {
    "stacked_2xgb_meta_lr": "#1b9e77", "XGBoost stack": "#1b9e77",
    "tabpfn": "#7570b3", "TabPFN": "#7570b3",
    "sequence": "#d95f02", "window-MLP": "#d95f02",
    "pooled_lr": "#666666",
}


def apply():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def box(ax, xy, w, h, text, fc="#eef3f8", ec="#3a6ea5", fontsize=8.5, weight="normal"):
    """Rounded text box centred at xy; returns (x, y, w, h) for arrow anchoring."""
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, zorder=2, fc=fc, ec=ec,
                                lw=1.2, boxstyle="round,pad=0.01,rounding_size=0.015"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3, weight=weight)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="#555555"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.1, color=color, zorder=1))


def save(fig, stem: Path) -> str:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"
