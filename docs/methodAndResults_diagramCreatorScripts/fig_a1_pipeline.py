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
    # Box title at the guideline axis-title size (10.5 pt).
    ax.text(cx, cy + h/2 - 0.32, title, ha="center", va="top",
            fontsize=10.5, weight="bold", color=S.INK)


def _chip(ax, cx, cy, w, h, label, color):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        fc=color, ec="white", lw=0.9, alpha=0.88))
    # Chip label at the guideline tick/legend size (8 pt).
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=8, weight="bold", color="white")


def main():
    S.apply()
    # Authored at the full-width landscape size (S.COL_WIDE, ~244 mm) so the
    # guideline point sizes (10.5 pt titles / 9 pt body / 8 pt chips) render with
    # clear padding inside every box; a column-width canvas cannot hold this many
    # stages at those sizes. The data canvas is symmetric about x = 11.8 (the
    # pipeline mid-point) and the limits hug the drawn ink, so bbox="tight" leaves
    # no surplus margin.
    fig, ax = plt.subplots(figsize=S.figsize("wide", 4.4))
    ax.set_xlim(0, 23.6); ax.set_ylim(0.2, 9.0); ax.axis("off")

    stage_y = 7.2
    stage_h = 3.0
    stage_w = 3.6
    # Even 4.9-unit spacing, symmetric about the canvas centre (11.8); the
    # 1.3-unit inter-box gap leaves clean room for the connecting arrows.
    stage_xs = [2.0, 6.9, 11.8, 16.7, 21.6]
    centre_x = stage_xs[2]
    arrow_pad = 0.18

    # ---- Stage 1: SHD diary source ------------------------------------------
    _stage_box(ax, stage_xs[0], stage_y, stage_w, stage_h, "Diary source")
    ax.text(stage_xs[0], stage_y - 0.25,
            "Park 2016 SHD\n62 analysed (63 patient_ids)\n4,516 patient-days",
            ha="center", va="center", fontsize=9, color=S.INK)

    # ---- Stage 2: Feature sets (4 colour-keyed chips, 2x2 grid) -------------
    _stage_box(ax, stage_xs[1], stage_y, stage_w, stage_h, "Feature sets")
    fset_specs = [
        ("full", "full", 52),
        ("no-roll", "no_rolling", 26),
        ("park", "park", 6),
    ]
    for j, (lbl, key, nfeat) in enumerate(fset_specs):
        cx = stage_xs[1] - 0.85 + 1.7 * (j % 2)
        cy = stage_y + 0.20 - 1.05 * (j // 2)
        _chip(ax, cx, cy, 1.55, 0.55, lbl, S.FEATURE_SET[key])
        ax.text(cx, cy - 0.44, f"{nfeat} feat",
                ha="center", va="top", fontsize=8, color=S.SOFT, style="italic")

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
        cy = stage_y + 0.20 - 1.05 * (j // 2)
        _chip(ax, cx, cy, 1.55, 0.55, lbl, S.SPLIT[key])
    ax.text(stage_xs[2], stage_y - 1.95,
            "70/30, 70/15/15, 80/20",
            ha="center", va="center", fontsize=8, color=S.INK, weight="bold")

    # ---- Stage 4: Architecture families -------------------------------------
    _stage_box(ax, stage_xs[3], stage_y, stage_w, stage_h, "Architectures")
    # Two-line chips: the "Add N" tag stacks above the architecture name so each
    # token keeps clear padding inside the chip border (a single line overflows).
    arch_specs = [
        ("Add 0\nXGB stack", "XGBoost stack"),
        ("Add 1\nTabPFN ×6", "TabPFN"),
        ("Add 4\nsequence ×3", "window-MLP"),
    ]
    for j, (lbl, key) in enumerate(arch_specs):
        cy = stage_y + 0.50 - 0.78 * j
        _chip(ax, stage_xs[3], cy, 3.0, 0.72, lbl, S.ARCH[key])

    # ---- Stage 5: Evaluation (output) ---------------------------------------
    _stage_box(ax, stage_xs[4], stage_y, stage_w, stage_h, "Evaluation",
               role="output")
    ax.text(stage_xs[4], stage_y - 0.32,
            "AUROC, AUPRC\ncalibration slope\nECE, Brier\n1000-it bootstrap",
            ha="center", va="center", fontsize=9, color=S.INK)

    # ---- Arrows between stages ----------------------------------------------
    # The stage names already carry the transition meaning, so the connectors
    # are kept label-free: a clean arrow in the gap reads better than text
    # crowded above the arrowhead.
    for i in range(4):
        x_start = stage_xs[i] + stage_w / 2 + arrow_pad
        x_end = stage_xs[i + 1] - stage_w / 2 - arrow_pad
        S.arrow(ax, (x_start, stage_y), (x_end, stage_y))

    # ---- Analysis + validation band below -----------------------------------
    band_y = 1.6
    band_h = 2.6
    # Shifted a touch right of the pipeline centre so the band sits under the
    # right-hand stages that actually feed it (the two down-arrows originate from
    # Architectures + Evaluation).
    band_half = 10.6
    band_center = centre_x + 0.5
    band_x_left = band_center - band_half
    band_x_right = band_center + band_half
    ax.add_patch(FancyBboxPatch(
        (band_x_left, band_y - band_h/2), band_x_right - band_x_left, band_h,
        boxstyle="round,pad=0.05,rounding_size=0.20",
        fc=S.PALE_FILL, ec=S.SOFT, lw=1.0))
    ax.text(band_center, band_y + band_h/2 - 0.34,
            "Analysis & validation layers (consume fitted models + predictions)",
            ha="center", va="top", fontsize=10.5, weight="bold", color=S.INK)

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
        ax.text(cx, band_y + 0.42, tag, ha="center", va="center",
                fontsize=10, weight="bold", color=S.INK)
        ax.text(cx, band_y - 0.02, name, ha="center", va="center",
                fontsize=9, color=S.INK)
        ax.text(cx, band_y - 0.62, methods, ha="center", va="center",
                fontsize=8, color=S.SOFT, style="italic")

    # ---- Arrows from architectures + evaluation down to the band ------------
    # Unlabelled: the band title already names the two inputs it consumes
    # ("fitted models + predictions"), so labelling each arrow would both repeat
    # that text and force the connector line through the label.
    S.arrow(ax, (stage_xs[3], stage_y - stage_h/2),
            (stage_xs[3] - 1.6, band_y + band_h/2))
    S.arrow(ax, (stage_xs[4], stage_y - stage_h/2),
            (stage_xs[4] - 0.8, band_y + band_h/2))

    # suptitle size inherits the 11.5 pt rcParam (no hardcode).
    fig.suptitle("Benchmark pipeline: source → features → splits → architectures → evaluation\n"
                 "Park 2016 SHD, 62 analysed (63 patient_ids), 4,516 patient-days",
                 y=0.99)
    print("saved", S.save(fig, HERE / "figures" / "fig_a1_pipeline"))


if __name__ == "__main__":
    main()
