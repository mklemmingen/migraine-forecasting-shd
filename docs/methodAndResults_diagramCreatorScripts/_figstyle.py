"""Shared matplotlib styling for the paper figures.

Consistent fonts/sizes, a fixed colour palette for targets and architectures, and
a save() that writes PNG (raster preview) + PDF (vector, for LaTeX) for every
figure. Imported by the fig_*.py diagram-creator scripts.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def save(fig, stem: Path) -> str:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"
