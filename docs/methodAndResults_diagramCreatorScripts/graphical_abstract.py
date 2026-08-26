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
import csv
import os
import sys
from pathlib import Path
import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from PIL import Image

HERE = Path(__file__).resolve().parent
ICON_SVG = HERE / "Diary_Lauterbach.svg"
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiment"))
import _style as S  # noqa: E402

# Shared by the discrimination and per-patient panels so equal AUROC values sit at equal
# heights in both, making them comparable by eye. Must span the hero's 0.890 CI top and
# the per-patient cloud's 0.318 floor.
SHARED_YLIM = (0.30, 0.95)
SHARED_YTICKS = [0.3, 0.5, 0.7, 0.9]

CI_BAND_WIDTH = 0.085               # widest point of a CI density, in x units
MIG_TEXT = "#C25100"  # 4.70:1 on white; S.target_color("migraine") is 3.87:1
GRID_X_FRAC = 0.588                 # grid origin as a fraction of the cohort panel,
GRID_Y_FRAC = 0.287                 # chosen so the snapped box keeps its old position
GRID_MID_FRAC = 0.810               # horizontal centre of the snapped grid
GRID_PITCH_PX = 7                   # day-grid cell-to-cell distance, whole pixels
GRID_CELL_PX = 6                    # filled square inside that pitch, 1 px gutter

# Numbers: single source of truth, traced to article.tex.
# within_pi is the 95% prediction interval across patients, est +- 1.96*sqrt(tau^2 + SE^2),
# from the published Paule-Mandel tau^2 in experiment/5/within_person_summary_cv_*.csv
# (migraine 0.001344, headache 0.008189) and the SE implied by the published CI. It is the
# quantity that reconciles this panel with the per-patient panel: the CI is how precisely the
# MEAN within-person C is known, the prediction interval is where an individual patient sits.
# Without it a reader compares a 0.066-wide CI band against a dot cloud spanning 0.32 to 0.84
# on the same axis and reads a contradiction that is not there.
MIGRAINE = {"pooled": 0.791, "pooled_ci": (0.544, 0.890),
            "within": 0.558,  "within_ci": (0.511, 0.604),
            "within_pi": (0.472, 0.643),
            "slope": 1.386,   "slope_ci": (0.40, 2.07)}
HEADACHE = {"pooled": 0.653, "pooled_ci": (0.555, 0.740),
            "within": 0.542,  "within_ci": (0.509, 0.575),
            "within_pi": (0.361, 0.722),
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

    ax.text(GRID_MID_FRAC, 0.735, "100 typical diary days", fontsize=7.4, color="#5f5f5f",
            transform=ax.transAxes, va="center", ha="center")
    gax = ax.inset_axes([0.30, 0.295, 1.0, 0.38])
    gax.set_xlim(0, 10); gax.set_ylim(0, 10)
    gax.set_aspect("equal"); gax.axis("off")
    n_mig, n_hea = 7, 23                      # per 100 days: 7.2% and 23.5%
    for i in range(100):
        col = mig_col if i < n_mig else (hea_col if i < n_hea else "#e3e3e3")
        # one data unit is GRID_PITCH_PX once main() snaps the box, so a cell of
        # CELL/PITCH lands on whole pixels and every square rasterises identically
        gax.add_patch(Rectangle((i % 10, 9 - i // 10),
                                GRID_CELL_PX / GRID_PITCH_PX,
                                GRID_CELL_PX / GRID_PITCH_PX, fc=col, ec="none",
                                antialiased=False))

    # main() snaps the grid to GRID_X_FRAC of this panel, so the legend and the
    # captions take that fraction directly rather than measuring a box that is
    # still pre-snap at this point.
    gx0 = GRID_X_FRAC

    legend = [(mig_col, "7  migraine"),
              (hea_col, "16  other headache"),
              ("#e3e3e3", "77  headache free")]
    for i, (col, lab) in enumerate(legend):
        yy = 0.195 - i * 0.070
        ax.add_patch(Rectangle((gx0, yy), 0.055, 0.045, fc=col, ec="none",
                               transform=ax.transAxes, clip_on=False))
        ax.text(gx0 + 0.085, yy + 0.021, lab, fontsize=7.4, color="#3d3d3d",
                transform=ax.transAxes, va="center", ha="left")

    ax.text(GRID_MID_FRAC, -0.030, "62 patients, 4,516 diary days", fontsize=7.6, color="#3d3d3d",
            transform=ax.transAxes, va="center", ha="center")
    ax.text(GRID_MID_FRAC, -0.105, "Park 2016, 2 Korean clinics", fontsize=7.6, color="#5f5f5f",
            transform=ax.transAxes, va="center", ha="center")
    return gax


def _load_pooled_replicates() -> dict:
    """Load the patient-cluster bootstrap AUROC replicates, if they are on disk.

    run_patient_cluster_bootstrap.py writes every resampled AUROC alongside the
    summary CSV, so the pooled bands can be drawn from the actual resampling
    distribution instead of a curve inferred from its two percentiles. Returns an
    empty dict when the file is absent, which drops the pooled bands back to the
    analytic shape.
    """
    d = REPO / "experiment" / "_eval" / "_special"
    files = sorted(d.glob("patient_cluster_auroc_replicates_*.csv"))
    if not files:
        return {}
    out: dict[tuple[str, str], list[float]] = {}
    with open(files[-1], newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault((row["target"], row["architecture"]), []).append(
                float(row["auroc"]))
    return {k: np.asarray(v) for k, v in out.items()}


def _ci_density(ax, xi, est, ci, color, side, zorder, reps=None):
    """Draw a confidence interval as a 90-degree-rotated density on the AUROC axis.

    The thin spine spans exactly the interval, so the CI is still read off the
    y-axis directly; the filled curve beside it shows where the mass sits.

    Where the replicates exist (the two pooled AUROCs, resampled by patient
    cluster) the curve is a kernel density over those replicates -- the actual
    resampling distribution the published interval was cut from, not a stand-in.
    The within-person C-statistics have no such distribution to plot: they are
    Hanley-McNeil variances pooled by Paule-Mandel random effects, and that
    estimator's interval is normal for the pooled mean by construction, so the
    normal is drawn there rather than implied. Both intervals are asymmetric
    about the estimate, so each half takes its own sigma from its own half-width.

    Migraine and headache are drawn on opposite sides because at the within-person
    end the two estimates differ by 0.016 and the intervals sit almost on top of
    each other; concentric shapes would read as one blob. Both use the same
    maximum width, so the widths stay comparable between the two targets.
    """
    lo, hi = ci
    yy = np.linspace(lo, hi, 240)
    if reps is not None and len(reps) > 50:
        # Anchor the replicates to the published interval before drawing them. The
        # band's extent must stay the number the paper prints; only its SHAPE comes
        # from the resampling distribution. For migraine, whose re-run reproduced the
        # published CI exactly, this map is the identity. For headache TabPFN it
        # absorbs the offset from re-running a GPU-fitted cell on CPU (AUROC 0.669
        # against a published 0.653), which shifts location without telling us the
        # shape is wrong. Affine, so the skew is carried over unchanged.
        r_lo, r_hi = np.percentile(reps, [2.5, 97.5])
        if r_hi > r_lo:
            reps = lo + (reps - r_lo) * (hi - lo) / (r_hi - r_lo)
        # Silverman bandwidth on the replicates
        sd = float(np.std(reps, ddof=1))
        iqr = float(np.subtract(*np.percentile(reps, [75, 25])))
        h = 0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * len(reps) ** (-0.2)
        dens = np.exp(-0.5 * ((yy[:, None] - reps[None, :]) / h) ** 2).sum(1)
    else:
        sd_lo = max((est - lo) / 1.96, 1e-6)
        sd_hi = max((hi - est) / 1.96, 1e-6)
        sd = np.where(yy < est, sd_lo, sd_hi)
        dens = np.exp(-0.5 * ((yy - est) / sd) ** 2)
    dens = dens / dens.max() * CI_BAND_WIDTH * side
    ax.fill_betweenx(yy, xi, xi + dens, fc=color, ec="none", alpha=0.28,
                     zorder=zorder)
    ax.plot(xi + dens, yy, color=color, lw=0.7, alpha=0.9, zorder=zorder + 0.1)
    ax.plot([xi, xi], [lo, hi], color=color, lw=0.9, alpha=0.55, zorder=zorder + 0.1)


def _chance_label(ax):
    """Label the 0.5 reference line identically in both discrimination panels.

    The blended transform fixes x as a fraction of the axes and y in data units, so
    the label sits at the same relative spot in the slopegraph and the per-patient
    panel even though their x-limits differ. Below the line rather than on it: the
    two within-person rules sit only 0.04 above chance, so the space overhead is
    crowded and the space beneath is clear.
    """
    ax.text(0.02, 0.5, "chance", fontsize=7.5, color="#5f5f5f", ha="left", va="top",
            transform=ax.get_yaxis_transform())


def _draw_slopegraph(ax) -> None:
    mig_col = S.target_color("migraine")
    hea_col = S.target_color("headache")
    # Both pooled bands must come from the same kind of object: an empirical curve
    # beside an analytic one would read as a difference between the targets rather
    # than a difference in what could be re-run. If either cell's replicates are
    # missing, both fall back to the analytic shape.
    reps = _load_pooled_replicates()
    if not {("migraine", "XGBoost"), ("headache", "TabPFN")} <= set(reps):
        reps = {}
    # x positions: pooled at 0, within at 1
    x = [0, 1]
    # Dashed reference at 0.5 (chance); the y-axis tick at 0.5 carries the
    # numeric and the dashed line carries the semantic, no inline label needed.
    ax.axhline(0.5, color=S.REF_COLOR, lw=0.8, ls="--", alpha=0.6, zorder=1)
    # Migraine slope
    mig_y = [MIGRAINE["pooled"], MIGRAINE["within"]]
    ax.plot(x, mig_y, color=mig_col, lw=2.2, zorder=4,
            marker="o", markersize=6.5, mfc=mig_col, mec="white", mew=1.0)
    for xi, yi, ci in [(0, MIGRAINE["pooled"], MIGRAINE["pooled_ci"]),
                       (1, MIGRAINE["within"], MIGRAINE["within_ci"])]:
        _ci_density(ax, xi, yi, ci, mig_col, -1, 2.0,
                    reps=reps.get(("migraine", "XGBoost")) if xi == 0 else None)
    # Headache slope
    hea_y = [HEADACHE["pooled"], HEADACHE["within"]]
    ax.plot(x, hea_y, color=hea_col, lw=2.2, zorder=4,
            marker="o", markersize=6.5, mfc=hea_col, mec="white", mew=1.0)
    for xi, yi, ci in [(0, HEADACHE["pooled"], HEADACHE["pooled_ci"]),
                       (1, HEADACHE["within"], HEADACHE["within_ci"])]:
        _ci_density(ax, xi, yi, ci, hea_col, +1, 2.0,
                    reps=reps.get(("headache", "TabPFN")) if xi == 0 else None)
    # Endpoint value labels: white bbox lifts the text off crossing slopes.
    # Within-person endpoints sit only 0.016 apart in y (migraine 0.558,
    # headache 0.542), which is below the 8.5pt label line-height at this
    # axes scale; the labels therefore need a small vertical offset (±0.04)
    # to avoid stacking on top of each other. clip_on=False lets labels
    # render into the inter-panel wspace if they exceed the data area.
    label_box = dict(fc="white", ec="none", pad=0.8)
    ax.text(-0.16, MIGRAINE["pooled"], f"{MIGRAINE['pooled']:.2f}", ha="right",
            va="center", fontsize=11, color=MIG_TEXT, fontweight="bold",
            bbox=label_box, clip_on=False)
    ax.text(1.13, MIGRAINE["within"] + 0.04, f"{MIGRAINE['within']:.2f}",
            ha="left", va="bottom", fontsize=11, color=MIG_TEXT, fontweight="bold",
            bbox=label_box, clip_on=False)
    ax.text(-0.16, HEADACHE["pooled"], f"{HEADACHE['pooled']:.2f}", ha="right",
            va="center", fontsize=11, color=hea_col, fontweight="bold",
            bbox=label_box, clip_on=False)
    ax.text(1.13, HEADACHE["within"] - 0.04, f"{HEADACHE['within']:.2f}",
            ha="left", va="top", fontsize=11, color=hea_col, fontweight="bold",
            bbox=label_box, clip_on=False)
    # Sits between the two within-person endpoint labels, where a reader who has just
    # looked at the per-patient panel would otherwise read this narrow band as the
    # spread across patients rather than the precision of their average.
    ax.text(1.13, (MIGRAINE["within"] + HEADACHE["within"]) / 2,
            "95% CI\nof the mean", fontsize=6.0, color="#7a7a7a", ha="left",
            va="center", linespacing=1.25, clip_on=False)

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
    _chance_label(ax)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(SHARED_YLIM)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pooled AUROC\n(all patients' days together)",
                        "within-person C-statistic\n(one patient's own days)"],
                       fontsize=8.5)
    ax.set_yticks(SHARED_YTICKS)
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

    # tie the panel to the hero: the same within-person C values the slopegraph lands on
    for tgt, d, col, txt in (("headache", HEADACHE, hea_col, hea_col),
                             ("migraine", MIGRAINE, mig_col, MIG_TEXT)):
        if not series.get(tgt):
            continue
        ax.axhline(d["within"], color=col, lw=1.2, alpha=0.85, zorder=2)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(SHARED_YLIM)
    ax.set_xticks([])
    ax.set_yticks(SHARED_YTICKS)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("per-patient AUROC", fontsize=8)
    n_mig, n_hea = len(series.get("migraine", [])), len(series.get("headache", []))
    ax.set_xlabel(f"one dot per patient\n({n_hea} headache patients, {n_mig} migraine)", fontsize=8)
    _chance_label(ax)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    S.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    fig, (ax_cohort, ax_gap, ax_cal) = plt.subplots(
        1, 3, figsize=(9.20, 3.00), dpi=100,
        gridspec_kw={"width_ratios": [1.0, 1.85, 1.15]},
    )
    grid_ax = _draw_cohort(ax_cohort)
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

    # A 10x10 grid only rasterises evenly if its box is an exact multiple of the
    # pitch and starts on a pixel boundary; otherwise the first cell loses a pixel.
    fig.canvas.draw()
    _fw, _fh = fig.get_size_inches() * fig.dpi
    _pb = ax_cohort.get_window_extent()
    _side = 10 * GRID_PITCH_PX
    _gx = round(_pb.x0 + GRID_X_FRAC * _pb.width)
    _gy = round(_pb.y0 + GRID_Y_FRAC * _pb.height)
    grid_ax.set_aspect("auto")
    grid_ax.set_axes_locator(None)      # inset locators override set_position
    grid_ax.set_position([_gx / _fw, _gy / _fh, _side / _fw, _side / _fh])

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
