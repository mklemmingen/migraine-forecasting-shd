"""Figure A1 - benchmark pipeline schematic.

The linear path from the open diary to the bootstrap evaluation, with the
post-hoc analysis and validation layers (Additions 2/3/5/6 and the
leave-one-site-out external validation) drawing on the fitted models and their
predictions. Each pipeline stage exposes its canonical sub-entities as small
colour-keyed chips so a reader can read which feature sets, splits, ratios and
architecture families exist without consulting the prose. Schematic only - no
data.

Usage: python fig_a1_pipeline.py
"""
# §11 compliance: schematic only (no headline metric).
#   §11.3 self-contained caption:        cohort name + n in suptitle
#   §11.6 CC BY 4.0 footer:              S.cc_by_footer() invoked
#   §11.1, §11.2, §11.5, §11.7:          N/A (no metric, no cell-specific cohort panel)
#   §11.10 no "substantial"/"large":     verified
#   §11.11 self-check:                   this block
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent


def _stage_box(ax, cx, cy, w, h, title, role="normal"):
    edge = S.BOX_EDGE.get(role, S.BOX_EDGE["normal"])
    lw = 1.6 if role == "output" else 1.2
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.20",
        fc=S.PALE_FILL, ec=edge, lw=lw))
    ax.text(cx, cy + h/2 - 0.30, title, ha="center", va="top",
            fontsize=9.5, weight="bold", color=S.INK)


def _chip(ax, cx, cy, w, h, label, color):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc=color, ec="white", lw=0.9, alpha=0.88))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=7.5, weight="bold", color="white")


def main():
    S.apply()
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.set_xlim(0, 22); ax.set_ylim(0, 10); ax.axis("off")

    stage_y = 7.0
    stage_h = 3.0
    stage_w = 3.6
    stage_xs = [2.0, 6.4, 11.0, 15.6, 20.0]
    arrow_pad = 0.15

    # ---- Stage 1: SHD diary source ------------------------------------------
    _stage_box(ax, stage_xs[0], stage_y, stage_w, stage_h, "Diary source")
    ax.text(stage_xs[0], stage_y - 0.20,
            "Park 2016 SHD\n63 patients\n4,516 patient-days",
            ha="center", va="center", fontsize=9, color=S.INK)

    # ---- Stage 2: Feature sets (4 colour-keyed chips, 2x2 grid) -------------
    _stage_box(ax, stage_xs[1], stage_y, stage_w, stage_h, "Feature sets")
    fset_specs = [
        ("full", "full", 52),
        ("spano", "spano", 31),
        ("no-roll", "no_rolling", 26),
        ("park", "park", 6),
    ]
    for j, (lbl, key, nfeat) in enumerate(fset_specs):
        cx = stage_xs[1] - 0.85 + 1.7 * (j % 2)
        cy = stage_y - 0.10 - 1.05 * (j // 2)
        _chip(ax, cx, cy, 1.45, 0.55, lbl, S.FEATURE_SET[key])
        ax.text(cx, cy - 0.45, f"{nfeat} features",
                ha="center", va="top", fontsize=6.8, color=S.SOFT, style="italic")

    # ---- Stage 3: Splits + ratios -------------------------------------------
    _stage_box(ax, stage_xs[2], stage_y, stage_w, stage_h, "Splits × ratios")
    split_specs = [
        ("chrono", "chrono"),
        ("strat", "stratified"),
        ("patient", "patient"),
        ("site", "site"),
    ]
    for j, (lbl, key) in enumerate(split_specs):
        cx = stage_xs[2] - 0.85 + 1.7 * (j % 2)
        cy = stage_y - 0.10 - 1.05 * (j // 2)
        _chip(ax, cx, cy, 1.45, 0.55, lbl, S.SPLIT[key])
    ax.text(stage_xs[2], stage_y - 2.45,
            "70/30 · 70/15/15 · 80/20",
            ha="center", va="center", fontsize=7.5, color=S.INK, weight="bold")

    # ---- Stage 4: Architecture families -------------------------------------
    _stage_box(ax, stage_xs[3], stage_y, stage_w, stage_h, "Architectures")
    arch_specs = [
        ("Add 0  XGB stack", "XGBoost stack"),
        ("Add 1  TabPFN × 6", "TabPFN"),
        ("Add 4  sequence × 3", "window-MLP"),
    ]
    for j, (lbl, key) in enumerate(arch_specs):
        cy = stage_y + 0.45 - 0.75 * j
        _chip(ax, stage_xs[3], cy, 2.9, 0.55, lbl, S.ARCH[key])

    # ---- Stage 5: Bootstrap evaluation (output) -----------------------------
    _stage_box(ax, stage_xs[4], stage_y, stage_w, stage_h, "Bootstrap eval",
               role="output")
    ax.text(stage_xs[4], stage_y - 0.30,
            "AUROC · AUPRC\ncalibration slope\nECE · Brier\n1000-it bootstrap",
            ha="center", va="center", fontsize=8.5, color=S.INK)

    # ---- Arrows between stages ----------------------------------------------
    arrow_labels = ["raw diary", "+ features", "+ splits", "predictions"]
    for i in range(4):
        x_start = stage_xs[i] + stage_w / 2 + arrow_pad
        x_end = stage_xs[i + 1] - stage_w / 2 - arrow_pad
        S.arrow(ax, (x_start, stage_y), (x_end, stage_y))
        ax.text((x_start + x_end) / 2, stage_y + 0.35, arrow_labels[i],
                ha="center", va="bottom", fontsize=7.5,
                color=S.SOFT, style="italic")

    # ---- Analysis + validation band below -----------------------------------
    band_y = 1.7
    band_h = 2.6
    band_x_left = 1.2
    band_x_right = 21.4
    ax.add_patch(FancyBboxPatch(
        (band_x_left, band_y - band_h/2), band_x_right - band_x_left, band_h,
        boxstyle="round,pad=0.05,rounding_size=0.20",
        fc=S.PALE_FILL, ec=S.SOFT, lw=1.0))
    ax.text((band_x_left + band_x_right) / 2, band_y + band_h/2 - 0.35,
            "Analysis & validation layers (consume fitted models + predictions)",
            ha="center", va="top", fontsize=9.5, weight="bold", color=S.INK)

    layer_labels = [
        ("Add 2", "explainability", "SHAP / ShapIQ / ALE"),
        ("Add 3", "temporal dependence", "ACF / Hawkes / Markov"),
        ("Add 5", "within-person", "C-statistic / leave-one-site-out"),
        ("Add 6", "clinical value", "decision curve / Brier skill"),
    ]
    n_layers = len(layer_labels)
    band_inner_left = band_x_left + 0.6
    band_inner_right = band_x_right - 0.6
    spacing = (band_inner_right - band_inner_left) / n_layers
    for j, (tag, name, methods) in enumerate(layer_labels):
        cx = band_inner_left + spacing * (j + 0.5)
        ax.text(cx, band_y + 0.35, tag, ha="center", va="center",
                fontsize=10, weight="bold", color=S.INK)
        ax.text(cx, band_y - 0.05, name, ha="center", va="center",
                fontsize=8.5, color=S.INK)
        ax.text(cx, band_y - 0.65, methods, ha="center", va="center",
                fontsize=7, color=S.SOFT, style="italic")

    # ---- Arrows from architectures + bootstrap eval down to the band --------
    S.arrow(ax, (stage_xs[3], stage_y - stage_h/2),
            (stage_xs[3] - 1.6, band_y + band_h/2))
    ax.text(stage_xs[3] - 1.0,
            (stage_y - stage_h/2 + band_y + band_h/2)/2 + 0.4,
            "fitted models", ha="center", va="center", fontsize=7.5,
            color=S.SOFT, style="italic")
    S.arrow(ax, (stage_xs[4], stage_y - stage_h/2),
            (stage_xs[4] - 0.8, band_y + band_h/2))
    ax.text(stage_xs[4] - 0.4,
            (stage_y - stage_h/2 + band_y + band_h/2)/2 + 0.4,
            "bootstrap predictions", ha="center", va="center", fontsize=7.5,
            color=S.SOFT, style="italic")

    fig.suptitle("Benchmark pipeline: source → features → splits → architectures → evaluation\n"
                 "Park 2016 SHD, n=62 patients, 4,516 patient-days",
                 fontsize=11, y=0.99)
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_a1_pipeline"))


if __name__ == "__main__":
    main()
