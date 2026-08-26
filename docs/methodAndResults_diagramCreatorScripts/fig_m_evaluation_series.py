"""Figures M1-M4 - one per evaluation question, schematic beside measured result.

Four short full-width figures rather than one composite: each row of the earlier
composite was already at the 2:1-3:1 aspect this literature uses for a full-width
explanatory figure, so stacking four of them was what compressed them. Split, each
sits in Methods where its metric is defined.

Panel (a) of each is schematic. Panel (b) is the canonical plot for that question,
drawn from the headline cells' own predictions rather than from summary numbers.
The prose explaining each schematic lives in the manuscript caption.

(a, b) Discrimination. The bold curve is the pooled ROC, its area the pooled AUROC.
The thin curves behind it are the same model scored inside each patient separately.
The pooled curve bows away from the diagonal; the per-patient curves lie along it.
That gap is the paper's central result, and it is a difference in shape, not a
difference in summary statistics.

(c) Probabilistic value. Brier skill against each patient's own recorded attack rate,
with its patient-cluster interval. Zero is the reference: a forecast on that line has
added nothing to what the patient's own rate already said.

(d) Clinical value. Decision-curve net benefit against the two default policies. A
model earns a decision only where its curve sits above both treat-all and treat-none.

Usage: python fig_m_evaluation_series.py   (needs figures/headline_predictions.npz)
"""
# §11 compliance: reports headline-cell metrics.
#   §11.1 metric + CI + n:               AUROC and skill carry intervals; n in the caption
#   §11.2 cell named:                    panel titles name target and architecture
#   §11.3 self-contained caption:        cohort named in the manuscript caption
#   §11.10 no "substantial"/"large":     verified
#   §11.11 self-check:                   this block
import csv
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiment"
CACHE = HERE / "figures" / "headline_predictions.npz"
MIN_POS = 5                      # the five-positive floor used throughout the paper


def _load_preds():
    z = np.load(CACHE, allow_pickle=True)
    return {t: (z[f"{t}_y"], z[f"{t}_p"], z[f"{t}_pid"]) for t in ("migraine", "headache")}


def _load_intervals():
    f = EXP / "_eval/_special/patient_cluster_bootstrap_20260530_125806.csv"
    rows = {(r["target"], r["architecture"]): r for r in csv.DictReader(open(f))}
    return rows[("migraine", "XGBoost")], rows[("headache", "TabPFN")]


def _net_benefit(y, p, thresholds):
    """Vickers net benefit: the model's, and the treat-everyone default."""
    n = len(y)
    prev = y.mean()
    nb_model, nb_all = [], []
    for t in thresholds:
        flag = p >= t
        tp = float(np.sum(flag & (y == 1)))
        fp = float(np.sum(flag & (y == 0)))
        w = t / (1.0 - t)
        nb_model.append(tp / n - (fp / n) * w)
        nb_all.append(prev - (1.0 - prev) * w)
    return np.array(nb_model), np.array(nb_all)


# The graphical abstract's scale, adopted verbatim so the two figures read as one
# family: headings 8.8 bold, values 11 bold, axis labels 8, ticks 7, body 7.4, and
# its three greys for body / secondary / tertiary text.
PT_HEAD, PT_VALUE, PT_AXIS, PT_TICK, PT_BODY, PT_FINE = 8.8, 11.0, 8.0, 7.0, 7.4, 6.8
BODY, SECOND, THIRD = "#3d3d3d", "#5f5f5f", "#7a7a7a"


def _lab(ax, x, y, t, size=PT_BODY, col=None, ha="left", weight="normal"):
    ax.text(x, y, t, fontsize=size, color=col or SECOND, ha=ha, va="center",
            fontweight=weight, linespacing=1.5)


def _sq(ax, x, y, side, col, alpha=1.0, anchor="center"):
    """A true square on a 0-100 box whose axes are not square.

    Every schematic here shares one coordinate space, but the four rows have
    different physical heights, so a Rectangle of equal width and height renders as a
    different rectangle in each. Deriving the height from the rendered box keeps a
    square square and, more importantly, keeps one schematic element the same size in
    every row.
    """
    bb = ax.get_window_extent()
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = side * (bb.width / xr) * (yr / bb.height)
    y0 = y if anchor == "bottom" else y - h / 2
    ax.add_patch(Rectangle((x, y0), side, h, fc=col, ec="none", alpha=alpha))
    return h


def _how_auroc(ax, att, hea):
    """AUROC counts ordered pairs: one attack day against one quiet day."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.annotate("", xy=(14, 92), xytext=(14, 18),
                arrowprops=dict(arrowstyle="-|>", color=S.GREY, lw=1.0))
    _lab(ax, 6, 55, "forecast risk", PT_FINE, SECOND, ha="center")
    for yy, col, lab in ((74, att, "attack day"), (34, "#cfcfcf", "quiet day")):
        _sq(ax, 26, yy, 11, col)
        _lab(ax, 42, yy, lab, PT_BODY, BODY)
    ax.annotate("", xy=(32, 66), xytext=(32, 42),
                arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.2))
    _lab(ax, 62, 54, "ranked\ncorrectly", PT_BODY, BODY, weight="bold")


def _how_within(ax, att):
    """The same pair test, confined to one patient's own diary."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.add_patch(Rectangle((4, 34), 52, 50, fc="#f4f4f4", ec="none"))
    _lab(ax, 8, 78, "one patient's diary", PT_FINE, SECOND)
    for yy, col in ((64, att), (42, "#cfcfcf")):
        _sq(ax, 24, yy, 11, col)
    # double-headed arrow beside the pair, long enough to read as a comparison
    ax.annotate("", xy=(16, 66), xytext=(16, 40),
                arrowprops=dict(arrowstyle="<|-|>", color=S.INK, lw=1.1,
                                mutation_scale=8))
    _lab(ax, 40, 53, "compared", PT_FINE, BODY)
    _sq(ax, 72, 58, 11, "#cfcfcf", alpha=0.5)
    ax.plot([70, 86], [66, 50], color=S.INK, lw=1.2)
    _lab(ax, 64, 40, "another patient's day:\nnever compared", PT_FINE, SECOND)


def _how_brier(ax, hea):
    """Brier compares the squared miss of the forecast with that of the patient's rate."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.plot([8, 92], [70, 70], color=S.GREY, lw=1.0)
    for x, t in ((8, "0"), (92, "1")):
        ax.plot([x, x], [68, 72], color=S.GREY, lw=1.0)
        _lab(ax, x, 63, t, PT_FINE, SECOND, ha="center")
    ax.plot([92], [70], "o", ms=6, mfc=S.INK, mec="none")
    _lab(ax, 92, 81, "what happened", PT_BODY, BODY, ha="right")
    for x, col, lab, side in ((66, hea, "forecast", 1), (36, "#9a9a9a", "own rate", -1)):
        ax.plot([x], [70], "o", ms=6, mfc="white", mec=col, mew=1.6)
        ax.plot([x, 92], [70, 70], color=col, lw=2.2, alpha=0.55,
                solid_capstyle="butt")
        sq = (92 - x) * 0.17
        _sq(ax, x, 34, sq, col, alpha=0.35, anchor="bottom")
        _lab(ax, x, 27, lab, PT_BODY, BODY)


def _how_dca(ax, att):
    """Net benefit trades attacks caught against unnecessary treatment."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    for i in range(6):
        _sq(ax, 10 + i * 8.5, 62, 7, att, anchor="bottom")
    _lab(ax, 10, 82, "attacks caught", PT_BODY, BODY)
    for i in range(4):
        _sq(ax, 10 + i * 8.5, 34, 7, "#cfcfcf", anchor="bottom")
    _lab(ax, 10, 46, "days treated for nothing", PT_BODY, BODY)
    _lab(ax, 58, 32, "weighted by how\nreluctant one is\nto treat", PT_FINE, SECOND)


def _skill_panel(ax, rows, cols):
    ax.axvline(0.0, color=S.INK, lw=1.0, zorder=3)
    for i, ((tgt, r), col) in enumerate(zip(rows, cols)):
        yy = 1 - i
        lo, hi = float(r["bs_cluster_lo"]), float(r["bs_cluster_hi"])
        ax.plot([lo, hi], [yy, yy], color=col, lw=2.6, solid_capstyle="round", zorder=4)
        ax.plot([float(r["brier_skill"])], [yy], "o", ms=7, mfc="white", mec=col,
                mew=2.0, zorder=5)
        _lab(ax, hi + 0.02, yy, tgt, PT_BODY, BODY)
    ax.annotate("no improvement", xy=(0.0, 1.62), xytext=(0.055, 1.70),
            fontsize=PT_FINE, color=SECOND, va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color=SECOND, lw=0.8))
    ax.set_ylim(-0.8, 1.9); ax.set_xlim(-0.32, 0.55)
    ax.set_yticks([]); ax.tick_params(axis="x", labelsize=PT_TICK, length=2)
    ax.set_xlabel("Brier skill against the patient's own attack rate", fontsize=PT_AXIS)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def _dca_panel(ax, preds, col):
    """Net benefit against the two default policies.

    Only migraine is drawn. Its cached predictions reproduce the published AUROC to
    four decimals; the headache TabPFN cell does not, because this machine has no GPU
    and CPU inference of that cell shifts the AUROC by 0.016 (see TODO H10). Drawing a
    curve from predictions that disagree with the reported number would put a shape
    under a claim the paper does not make, so headache's published reading is stated
    instead of drawn.
    """
    ts = np.linspace(0.01, 0.50, 120)
    ax.axhline(0.0, color=S.INK, lw=1.0, zorder=3)
    y, p, _ = preds["migraine"]
    nb, nb_all = _net_benefit(y, p, ts)
    ax.plot(ts, nb_all, color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.7, zorder=4)
    ax.plot(ts, nb, color=col, lw=2.0, zorder=5, solid_capstyle="round")
    j = int(len(ts) * 0.34)
    _lab(ax, ts[j], nb[j] + 0.007, "migraine", PT_BODY, col, weight="bold")
    _lab(ax, 0.505, 0.0, "treat none", PT_FINE, BODY)
    _lab(ax, 0.20, -0.055, "dashed: treat everyone", PT_FINE, SECOND)
    ax.set_xlim(0.01, 0.50); ax.set_ylim(-0.065, 0.105)
    ax.set_xlabel("threshold probability", fontsize=PT_AXIS)
    ax.set_ylabel("net benefit", fontsize=PT_AXIS)
    ax.tick_params(labelsize=PT_TICK, length=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _question(fig, gs, row, n, question, gloss):
    """Row header: the question this row answers, on the four-question spine."""
    ax = fig.add_subplot(gs[row, :]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.add_patch(Rectangle((0, 4), 4.6, 42, fc=S.INK, ec="none"))
    ax.text(1.1, 25, str(n), fontsize=PT_BODY, fontweight="bold", color="white",
            ha="center", va="center")
    ax.text(6.5, 34, question, fontsize=PT_HEAD, fontweight="bold", color=S.INK,
            ha="left", va="center")
    ax.text(6.5, 8, gloss, fontsize=PT_BODY, color=SECOND, ha="left", va="center")


def _panel_letter(ax, letter):
    ax.text(-0.02, 1.06, f"({letter})", fontsize=PT_HEAD, fontweight="bold",
            color=S.INK, transform=ax.transAxes, ha="left", va="bottom")


def _one(name, how, how_args, draw_right, right_args):
    """One figure in the series: the schematic beside the quantity it explains.

    Each is full column width and short, the shape this literature uses for a
    full-width explanatory figure. The prose that used to sit under each schematic
    now lives in the manuscript caption, which is where a journal expects it.
    """
    fig = plt.figure(figsize=(183 / 25.4, 62 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.62, 1.0],
                          left=0.055, right=0.985, top=0.88, bottom=0.185,
                          wspace=0.30)
    axl, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axl.set_xlim(0, 100); axl.set_ylim(0, 100); axl.axis("off")
    fig.canvas.draw()
    _panel_letter(axl, "a"); _panel_letter(axr, "b")
    how(axl, *how_args)
    draw_right(axr, *right_args)
    out = S.save(fig, HERE / "figures" / name)
    print("saved", Path(out).name)


def main() -> None:
    S.apply()
    import graphical_abstract as GA        # the abstract's own diagrams, reused as drawn
    preds = _load_preds()
    mig_row, hea_row = _load_intervals()
    ORA, BLU = S.target_color("migraine"), S.target_color("headache")

    # Only predictions that reproduce their published metric may be drawn as curves.
    y, p, _ = preds["migraine"]
    assert abs(roc_auc_score(y, p) - float(mig_row["auroc"])) < 5e-4, \
        "migraine cached predictions disagree with the published AUROC"

    _one("fig_m1_pooled_auroc", _how_auroc, (ORA, BLU), GA._draw_slopegraph, ())
    _one("fig_m2_within_person", _how_within, (ORA,), GA._draw_per_patient, ())
    _one("fig_m3_brier_skill", _how_brier, (BLU,), _skill_panel,
         ([("migraine", mig_row), ("headache", hea_row)], [ORA, BLU]))
    _one("fig_m4_net_benefit", _how_dca, (ORA,), _dca_panel, (preds, ORA))


if __name__ == "__main__":
    main()
