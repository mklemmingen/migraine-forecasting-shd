"""Graphical abstract for the JHP submission.

JHP specification: 920 x 300 px, <=150 KB, JPG/PNG/SVG, file name "Graphical Abstract"
(https://thejournalofheadacheandpain.biomedcentral.com/graphical-abstracts).

Built as a multi-panel scientific figure on the shared experiment/_style.py
conventions, so it matches the paper's other figures. A rework that hand-placed
text on a blank canvas was rejected: dropping the axes made it read as an
infographic rather than a figure.

Three zones, left to right:
  Cohort (text): participants, event rates, setting, model family
  Discrimination: paired slopegraph, pooled AUROC -> within-person C, one slope
    per outcome, Delta labelled, CIs at every endpoint, chance reference at 0.5
  Calibration: slope dotplot with CIs against the unity reference
  Bottom strip: the load-bearing claim, plus the near-chance and
    medication-overuse consequences

The decision-curve panel was removed: its curve was interpolated between six anchor
values rather than plotted from the real arrays, and after the medication-overuse
argument was retracted its message became a conditional ("net benefit exceeds
treat-all only at higher thresholds") that cannot be read at panel width. That
conclusion is carried in the bottom strip as text instead. TRIPOD+AI Item 8 is
satisfied by the paper's own figures, not by this submission asset.

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
import io

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from PIL import Image

HERE = Path(__file__).resolve().parent
ICON_SVG = HERE / "Diary_Lauterbach.svg"
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


def _load_diary_icon():
    """Rasterise Diary_Lauterbach.svg and crop to its ink bounding box.

    The SVG carries a wide white margin; drawn uncropped at panel scale it
    shrinks to an unreadable box-in-box. Cropping lets it read as a notebook.
    """
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    drawing = svg2rlg(str(ICON_SVG))
    drawing.scale(3.0, 3.0)
    drawing.width *= 3.0
    drawing.height *= 3.0
    im = Image.open(io.BytesIO(renderPM.drawToString(drawing, fmt="PNG"))).convert("RGB")
    grey = im.convert("L")
    bbox = grey.point(lambda v: 255 if v < 250 else 0).getbbox()
    return im.crop(bbox) if bbox else im


def _draw_cohort(ax) -> None:
    """Cohort panel: the diary itself, then 100 days as recorded.

    The day grid replaces text-only event rates. The base-rate imbalance is what
    drives the pooled-versus-within-person story, so it earns a visual.

    Counts are measured from data/processed/{headache,migraine}/diary.parquet over
    the 4,516-day analytic cohort, not taken from prose: headache 1,060 days
    (23.5%), migraine 325 (7.2%). Per 100 days that is 7 migraine, 16 other
    headache, 77 headache free. An earlier revision used 17/76 from a stated 24%
    headache rate, which is the raw-diary figure (1,099/4,579), not the analytic
    cohort's.
    """
    mig_col = S.target_color("migraine")
    hea_col = S.target_color("headache")
    ax.axis("off")

    iax = ax.inset_axes([0.0, 0.735, 0.245, 0.265])
    iax.imshow(_load_diary_icon(), aspect="equal")
    iax.axis("off")
    ax.text(0.30, 0.875, "Headache diary", fontsize=8.8, fontweight="bold",
            color=S.INK, transform=ax.transAxes, va="center", ha="left")

    ax.text(0.80, 0.735, "100 typical diary days", fontsize=7.4, color="#5f5f5f",
            transform=ax.transAxes, va="center", ha="center")
    gax = ax.inset_axes([0.30, 0.295, 1.0, 0.38])
    gax.set_xlim(0, 10); gax.set_ylim(0, 10)
    gax.set_aspect("equal"); gax.axis("off")
    n_mig, n_hea = 7, 23                      # per 100 days: 7.2% and 23.5%
    for i in range(100):
        col = mig_col if i < n_mig else (hea_col if i < n_hea else "#e3e3e3")
        gax.add_patch(Rectangle((i % 10 + 0.08, 9 - i // 10 + 0.08), 0.84, 0.84,
                                fc=col, ec="none"))

    # The grid is aspect-equal, so it sits centred inside its full-width inset;
    # caption and legend are centred on it rather than flush to the panel edge.
    ax.figure.canvas.draw()
    pbox, gbox = ax.get_window_extent(), gax.get_window_extent()
    gx0 = (gbox.x0 - pbox.x0) / pbox.width

    legend = [(mig_col, "7  migraine"),
              (hea_col, "16  other headache"),
              ("#e3e3e3", "77  headache free")]
    for i, (col, lab) in enumerate(legend):
        yy = 0.195 - i * 0.070
        ax.add_patch(Rectangle((gx0, yy), 0.055, 0.045, fc=col, ec="none",
                               transform=ax.transAxes, clip_on=False))
        ax.text(gx0 + 0.085, yy + 0.021, lab, fontsize=7.4, color="#3d3d3d",
                transform=ax.transAxes, va="center", ha="left")

    ax.text(0.80, -0.030, "62 patients, 4,516 diary days", fontsize=7.6, color="#3d3d3d",
            transform=ax.transAxes, va="center", ha="center")
    ax.text(0.80, -0.105, "Park 2016, 2 Korean clinics", fontsize=7.6, color="#5f5f5f",
            transform=ax.transAxes, va="center", ha="center")


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
    ax.set_xticklabels(["pooled AUROC\n(all patients' days together)",
                        "within-person C-statistic\n(one patient's own days)"],
                       fontsize=8.5)
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


PER_PATIENT_CSV = HERE / "figures" / "per_patient_auroc.csv"


def _draw_per_patient(ax) -> None:
    """Per-patient within-person AUROC, one dot per patient.

    This is the paper's finding shown at the level the clinical claim is made:
    individual patients scattered across the chance line, with the pooled value
    far above nearly all of them. It replaces the calibration panel, whose
    interval (0.401-2.067 for migraine) is too wide to support any reading.

    Values come from figures/per_patient_auroc.csv, written by
    export_per_patient_auroc.py, which refuses to emit unless the recomputed
    within-person C reproduces the published one.
    """
    import csv
    mig_col = S.target_color("migraine")
    hea_col = S.target_color("headache")
    series = {}
    with open(PER_PATIENT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            series.setdefault(row["target"], []).append(float(row["auroc"]))

    ax.axhline(0.5, color=S.REF_COLOR, lw=0.9, ls="--", alpha=0.7, zorder=1)
    for tgt, col in (("headache", hea_col), ("migraine", mig_col)):
        vals = sorted(series.get(tgt, []))
        if not vals:
            continue
        # normalise rank so the two targets (different k) share one axis
        x = [i / (len(vals) - 1) for i in range(len(vals))] if len(vals) > 1 else [0.5]
        ax.plot(x, vals, "o", ms=3.4, color=col, alpha=0.85, mec="none", zorder=3)

    # tie the panel to the hero: the same within-person C the slopegraph lands on
    if series.get("migraine"):
        ax.axhline(MIGRAINE["within"], color=mig_col, lw=1.2, alpha=0.85, zorder=2)
        ax.text(1.05, MIGRAINE["within"], f"{MIGRAINE['within']:.2f}", fontsize=7.5,
                color=MIG_TEXT, va="center", ha="left", fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.9))
    ax.set_xlim(-0.06, 1.20)
    ax.set_ylim(0.36, 0.84)
    ax.set_xticks([])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("per-patient AUROC", fontsize=8)
    n_mig = len(series.get("migraine", []))
    ax.set_xlabel(f"one dot per patient\n(migraine, n = {n_mig})", fontsize=8)
    ax.text(-0.04, 0.5, "chance", fontsize=7.5, color="#5f5f5f", va="bottom", ha="left")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    S.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    fig, (ax_cohort, ax_gap, ax_cal) = plt.subplots(
        1, 3, figsize=(9.21, 3.00), dpi=100,
        gridspec_kw={"width_ratios": [1.0, 1.85, 1.15]},
    )
    _draw_cohort(ax_cohort)
    _draw_slopegraph(ax_gap)
    if PER_PATIENT_CSV.exists():
        _draw_per_patient(ax_cal)
    else:
        print('  per_patient_auroc.csv absent; keeping the calibration panel')
        _draw_calibration(ax_cal)

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
        (" on next-day forecasting.", S.INK, "bold"),
    ]
    line2 = [
        ("Within-person forecasting is near chance for migraine and headache. The "
         "all-patient versus single-patient gap reflects how much patients differ in "
         "attack frequency.",
         "#3d3d3d", "normal"),
    ]
    line3 = [
        ("Benchmarked: XGBoost stack, TabPFN, and sequence baselines "
         "(window-MLP, GRU, 1D-CNN).", "#7a7a7a", "normal"),
    ]
    line4 = [
        ("Headline cell: same-day diary plus engineered history features, "
         "chronological 70/30 split; migraine XGBoost, headache TabPFN.",
         "#7a7a7a", "normal"),
    ]
    rows = []
    for segs, size in ((line1, 8.2), (line2, 7.8), (line3, 6.8), (line4, 6.8)):
        rows.append(HPacker(align="baseline", pad=0, sep=0, children=[
            TextArea(t, textprops=dict(color=c, fontsize=size, fontweight=w))
            for t, c, w in segs]))
    hpacker = VPacker(children=rows, align="center", pad=0, sep=3)
    ab = AnnotationBbox(hpacker, (0.5, 0.025), xycoords="figure fraction",
                        frameon=False, box_alignment=(0.5, 0))
    fig.add_artist(ab)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.965, bottom=0.355,
                        wspace=0.55)

    # only the per-patient panel moves left; the hero keeps its width
    _b = ax_cal.get_position()
    ax_cal.set_position([_b.x0 - 0.022, _b.y0, _b.width, _b.height])

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
