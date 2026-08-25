"""Graphical abstract for the JHP submission.

JHP specification: 920 x 300 px, <=150 KB, JPG/PNG/SVG, file name "Graphical Abstract"
(https://thejournalofheadacheandpain.biomedcentral.com/graphical-abstracts).

Built as a multi-panel scientific figure on the shared experiment/_style.py
conventions, so it matches the paper's other figures. A rework that hand-placed
text on a blank canvas was rejected: dropping the axes made it read as an
infographic rather than a figure.

Four zones, left to right:
  Cohort (text): participants, event rates, setting, model family
  Discrimination: paired slopegraph, pooled AUROC -> within-person C, one slope
    per outcome, Delta labelled, CIs at every endpoint, chance reference at 0.5
  Calibration: slope dotplot with CIs against the unity reference
  Clinical utility: decision-curve sparkline, clinically plausible low-threshold
    band shaded
  Bottom strip: the load-bearing claim, plus the near-chance and
    medication-overuse consequences

TRIPOD+AI for Abstracts coverage: Item 1 (title/outcome/population in the bottom
strip), Item 5 (setting), Item 8 (discrimination + calibration + clinical utility
triad), Item 9 (n + patient-days), Item 11 (CIs on every point estimate), Item 12
(interpretation in the claim sentence).

Two constraints that are correctness, not taste:
  * Orange TEXT uses MIG_TEXT (#C25100, 4.70:1 on white). The palette orange
    #D55E00 is 3.87:1 and fails the 4.5:1 WCAG floor; it stays on marks and lines.
  * CIs are patient-CLUSTER, not patient-day. Table 2 of article.tex.

Numbers trace to article.tex and supplementary_body.tex; see
graphical_abstract_review.md in the paper working tree before changing any value.

Usage: python graphical_abstract.py
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiment"))
import _style as S  # noqa: E402

MIG_TEXT = "#C25100"  # 4.70:1 on white; S.target_color("migraine") is 3.87:1

# Numbers — single source of truth, traced to body sections.
MIGRAINE = {"pooled": 0.791, "pooled_ci": (0.544, 0.890),
            "within": 0.558,  "within_ci": (0.511, 0.604),
            "slope": 1.386,   "slope_ci": (0.40, 2.07)}
HEADACHE = {"pooled": 0.653, "pooled_ci": (0.555, 0.740),
            "within": 0.542,  "within_ci": (0.509, 0.575),
            "slope": 1.094,   "slope_ci": (0.594, 1.429)}


def _draw_cohort(ax) -> None:
    """Cohort, setting and model family as plain text. No icon: the previous
    diary glyph rendered as an uninformative box-in-box."""
    ax.axis("off")
    lines = [("Cohort", True),
             ("62 patients, 4,516 patient-days", False),
             ("migraine on 7.2% of days", False),
             ("headache on 24% of days", False),
             ("", False),
             ("Setting", True),
             ("Park 2016 smartphone diary,", False),
             ("2 Korean clinics", False),
             ("", False),
             ("Models", True),
             ("TabPFN, XGBoost, window-MLP", False)]
    yy = 0.98
    for txt, is_head in lines:
        if not txt:
            yy -= 0.034
            continue
        ax.text(0.0, yy, txt, transform=ax.transAxes, fontsize=8.2,
                fontweight="bold" if is_head else "normal",
                color=S.INK if is_head else "#3d3d3d", va="top", ha="left")
        yy -= 0.086


def _draw_slopegraph(ax) -> None:
    mig_col = S.target_color("migraine")
    hea_col = S.target_color("headache")
    # x positions: pooled at 0, within at 1
    x = [0, 1]
    # Dashed reference at 0.5 (chance); the y-axis tick at 0.5 carries the
    # numeric and the dashed line carries the semantic, no inline label needed.
    ax.axhline(0.5, color=S.REF_COLOR, lw=0.8, ls="--", alpha=0.6, zorder=1)
    # Migraine slope
    mig_y = [MIGRAINE["pooled"], MIGRAINE["within"]]
    ax.plot(x, mig_y, color=mig_col, lw=2.2, zorder=4,
            marker="o", markersize=8, mfc=mig_col, mec=S.INK, mew=0.8)
    # Migraine CI whiskers
    for xi, yi, ci in [(0, MIGRAINE["pooled"], MIGRAINE["pooled_ci"]),
                       (1, MIGRAINE["within"], MIGRAINE["within_ci"])]:
        ax.plot([xi, xi], [ci[0], ci[1]], color=mig_col, lw=1.2, alpha=0.7, zorder=3)
    # Headache slope
    hea_y = [HEADACHE["pooled"], HEADACHE["within"]]
    ax.plot(x, hea_y, color=hea_col, lw=2.2, zorder=4,
            marker="o", markersize=8, mfc=hea_col, mec=S.INK, mew=0.8)
    for xi, yi, ci in [(0, HEADACHE["pooled"], HEADACHE["pooled_ci"]),
                       (1, HEADACHE["within"], HEADACHE["within_ci"])]:
        ax.plot([xi, xi], [ci[0], ci[1]], color=hea_col, lw=1.2, alpha=0.7, zorder=3)
    # Endpoint value labels
    # Endpoint value labels — white bbox lifts the text off crossing slopes.
    # Within-person endpoints sit only 0.016 apart in y (migraine 0.558,
    # headache 0.542), which is below the 8.5pt label line-height at this
    # axes scale; the labels therefore need a small vertical offset (±0.04)
    # to avoid stacking on top of each other. clip_on=False lets labels
    # render into the inter-panel wspace if they exceed the data area.
    label_box = dict(fc="white", ec="none", pad=0.8)
    ax.text(-0.08, MIGRAINE["pooled"], f"{MIGRAINE['pooled']:.2f}", ha="right",
            va="center", fontsize=11, color=MIG_TEXT, fontweight="bold",
            bbox=label_box, clip_on=False)
    ax.text(1.06, MIGRAINE["within"] + 0.04, f"{MIGRAINE['within']:.2f}",
            ha="left", va="bottom", fontsize=11, color=MIG_TEXT, fontweight="bold",
            bbox=label_box, clip_on=False)
    ax.text(-0.08, HEADACHE["pooled"], f"{HEADACHE['pooled']:.2f}", ha="right",
            va="center", fontsize=11, color=hea_col, fontweight="bold",
            bbox=label_box, clip_on=False)
    ax.text(1.06, HEADACHE["within"] - 0.04, f"{HEADACHE['within']:.2f}",
            ha="left", va="top", fontsize=11, color=hea_col, fontweight="bold",
            bbox=label_box, clip_on=False)
    # Delta annotations at midpoint
    mig_delta = MIGRAINE["pooled"] - MIGRAINE["within"]
    hea_delta = HEADACHE["pooled"] - HEADACHE["within"]
    ax.text(0.5, (MIGRAINE["pooled"] + MIGRAINE["within"]) / 2 + 0.04,
            f"Δ {mig_delta:.2f}", ha="center", va="bottom",
            fontsize=9.5, color=MIG_TEXT, fontweight="bold")
    ax.text(0.5, (HEADACHE["pooled"] + HEADACHE["within"]) / 2 - 0.04,
            f"Δ {hea_delta:.2f}", ha="center", va="top",
            fontsize=9, color=hea_col, fontweight="bold")
    # Target identity is carried by colour + the bottom-strip claim sentence
    # (which names "migraine" and "headache" explicitly); no in-panel target
    # word labels here.
    ax.text(-0.22, 0.5, "chance", fontsize=7.5, color="#5f5f5f", ha="left",
            va="center", bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.36, 0.95)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pooled\nAUROC", "within-person\nC-statistic"], fontsize=8.5)
    ax.set_yticks([0.5, 0.7, 0.9])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("discrimination", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_calibration(ax) -> None:
    mig_col = S.target_color("migraine")
    hea_col = S.target_color("headache")
    # Dashed reference at 1.0; the y-axis tick already carries the numeric
    # and "calibration slope" axis label tells the reader 1.0 is the ideal.
    ax.axhline(1.0, color=S.REF_COLOR, lw=0.8, ls="--", alpha=0.6, zorder=1)
    # Dots with CIs
    for i, (name, d, col) in enumerate(
        [("migraine", MIGRAINE, mig_col), ("headache", HEADACHE, hea_col)]
    ):
        x = i
        ax.errorbar([x], [d["slope"]],
                    yerr=[[d["slope"] - d["slope_ci"][0]],
                          [d["slope_ci"][1] - d["slope"]]],
                    fmt="o", color=col, ecolor=col, elinewidth=1.2,
                    capsize=3, markersize=7, mfc=col, mec=S.INK, mew=0.7,
                    zorder=4)
        # Value label to the right of each dot, vertically centred on the
        # point estimate, so it does not float at the top of the panel.
        ax.text(x + 0.15, d["slope"], f"{d['slope']:.2f}", ha="left",
                va="center", fontsize=8, color=col, fontweight="bold")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 2.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mig.", "hea."], fontsize=8)
    ax.set_yticks([0, 1, 2])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("calibration slope", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_dca_sparkline(ax) -> None:
    mig_col = S.target_color("migraine")
    hea_col = S.target_color("headache")
    # Anchor values traced from fig_d3_decision_curve.py headline-cell output
    # at the canonical (full_features, chrono, 70/30, TabPFN) cells; the
    # sparkline is a linear interpolation between these anchors so the curve
    # shape is data-bound rather than fabricated. Source: fig_d3 run log,
    # bodysect 3.7.
    t_anchor = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50])
    headache_anchor = np.array([0.18, 0.152, 0.102, 0.052, 0.031, 0.016])
    migraine_anchor = np.array([0.045, 0.033, 0.020, 0.015, 0.007, -0.002])
    t = np.linspace(0.01, 0.50, 50)
    headache_nb = np.interp(t, t_anchor, headache_anchor)
    migraine_nb = np.interp(t, t_anchor, migraine_anchor)
    # Zero reference (treat none)
    ax.axhline(0.0, color=S.REF_COLOR, lw=0.6, ls=":", alpha=0.6, zorder=1)
    # Highlight clinically plausible sub-band t in [0.01, 0.10]
    ax.axvspan(0.01, 0.10, color="grey", alpha=0.10, zorder=0)
    # Lines
    ax.plot(t, headache_nb, color=hea_col, lw=2.0, zorder=3)
    ax.plot(t, migraine_nb, color=mig_col, lw=2.0, zorder=3)
    # Endpoint dots and labels
    ax.plot(t[0], headache_nb[0], "o", color=hea_col,
            markersize=6, mfc=hea_col, mec=S.INK, mew=0.7, zorder=4)
    ax.plot(t[0], migraine_nb[0], "o", color=mig_col,
            markersize=6, mfc=mig_col, mec=S.INK, mew=0.7, zorder=4)
    # Lift the endpoint labels well above the curves with a white bbox so the
    # curve strokes are not visually broken.
    label_box = dict(fc="white", ec="none", pad=0.8)
    ax.text(0.48, headache_nb[-1] + 0.025, "hea.", color=hea_col,
            fontsize=7.5, fontweight="bold", ha="right", va="bottom",
            bbox=label_box)
    ax.text(0.48, migraine_nb[-1] - 0.025, "mig.", color=mig_col,
            fontsize=7.5, fontweight="bold", ha="right", va="top",
            bbox=label_box)
    ax.set_xlim(0.01, 0.50)
    ax.set_ylim(-0.05, 0.20)
    ax.set_xticks([0.10, 0.50])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_yticks([0.0, 0.10, 0.20])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("threshold p", fontsize=8)
    ax.set_ylabel("net benefit", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    S.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    fig, (ax_cohort, ax_gap, ax_cal, ax_dca) = plt.subplots(
        1, 4, figsize=(9.21, 3.00), dpi=100,
        gridspec_kw={"width_ratios": [1.05, 1.6, 1.15, 1.25]},
    )
    _draw_cohort(ax_cohort)
    _draw_slopegraph(ax_gap)
    _draw_calibration(ax_cal)
    _draw_dca_sparkline(ax_dca)

    # Bottom-strip claim sentence (TRIPOD+AI for Abstracts Items 1 + 12).
    # The target words carry their data colour so the sentence itself serves
    # as the colour legend; HPacker keeps the colored segments baseline-aligned.
    mig_col = S.target_color("migraine")
    hea_col = S.target_color("headache")
    line1 = [
        ("Pooled AUROC overstates within-person discrimination by 0.23 ",
         S.INK, "bold"),
        ("(migraine)", MIG_TEXT, "bold"),
        (" and 0.11 ", S.INK, "bold"),
        ("(headache)", hea_col, "bold"),
        (" on next-day forecasting, Park 2016 Korean SHD.", S.INK, "bold"),
    ]
    line2 = [
        ("Within-person forecasting is near chance; at plausible thresholds the "
         "model alarms on 98 to 100% of days, above ICHD-3 overuse ceilings.",
         "#3d3d3d", "normal"),
    ]
    rows = []
    for segs, size in ((line1, 8.2), (line2, 7.8)):
        rows.append(HPacker(align="baseline", pad=0, sep=0, children=[
            TextArea(t, textprops=dict(color=c, fontsize=size, fontweight=w))
            for t, c, w in segs]))
    hpacker = VPacker(children=rows, align="center", pad=0, sep=3)
    ab = AnnotationBbox(hpacker, (0.5, 0.025), xycoords="figure fraction",
                        frameon=False, box_alignment=(0.5, 0))
    fig.add_artist(ab)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.95, bottom=0.235,
                        wspace=0.62)

    out_png = HERE / "figures" / "graphical_abstract.png"
    fig.savefig(out_png, dpi=100, bbox_inches=None,
                facecolor="white", edgecolor="none")
    plt.close(fig)

    with Image.open(out_png) as im:
        if im.size != (920, 300):
            im = im.resize((920, 300), Image.LANCZOS)
            im.save(out_png, optimize=True)
        w, h = im.size

    size_kb = os.path.getsize(out_png) / 1024.0
    print(f"saved {out_png.name}  {w}x{h} px, {size_kb:.1f} KB")
    print(f"  JHP target: 920x300 px, <= 150 KB  ->  "
          f"{'PASS' if (w == 920 and h == 300 and size_kb <= 150) else 'CHECK'}")


if __name__ == "__main__":
    main()
