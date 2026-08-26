"""Figure A6 - the evaluation argument in three forms.

Each panel is the canonical plot for the question it answers, drawn from the headline
cells' own predictions rather than from summary numbers.

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

Usage: python fig_a6_pooled_vs_within.py   (needs figures/headline_predictions.npz)
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


def _how_auroc(ax, att, hea):
    """AUROC counts ordered pairs: one attack day against one quiet day."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    _lab(ax, 0, 95, "how it is counted", PT_BODY, BODY, weight="bold")
    ax.annotate("", xy=(14, 88), xytext=(14, 22),
                arrowprops=dict(arrowstyle="-|>", color=S.GREY, lw=1.0))
    _lab(ax, 8, 55, "forecast risk", PT_FINE, SECOND, ha="center")
    ax.get_figure().canvas.draw()
    for yy, col, lab in ((74, att, "attack day"), (34, "#cfcfcf", "quiet day")):
        ax.add_patch(Rectangle((26, yy - 6), 12, 12, fc=col, ec="none"))
        _lab(ax, 42, yy, lab, PT_BODY, BODY)
    ax.annotate("", xy=(32, 66), xytext=(32, 42),
                arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.2))
    _lab(ax, 62, 54, "ranked\ncorrectly", PT_BODY, BODY, weight="bold")
    _lab(ax, 0, 10, "AUROC is the share of all attack / quiet\npairs the model ranks this "
                    "way. 0.50 is chance.", PT_FINE, THIRD)


def _how_within(ax, att):
    """The same pair test, confined to one patient's own diary."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    _lab(ax, 0, 95, "how it is counted", PT_BODY, BODY, weight="bold")
    ax.add_patch(Rectangle((4, 34), 52, 50, fc="#f4f4f4", ec="none"))
    _lab(ax, 8, 78, "one patient's diary", PT_FINE, SECOND)
    for yy, col in ((64, att), (42, "#cfcfcf")):
        ax.add_patch(Rectangle((24, yy - 6), 12, 12, fc=col, ec="none"))
    # double-headed arrow beside the pair, long enough to read as a comparison
    ax.annotate("", xy=(16, 66), xytext=(16, 40),
                arrowprops=dict(arrowstyle="<|-|>", color=S.INK, lw=1.1,
                                mutation_scale=8))
    _lab(ax, 40, 53, "compared", PT_FINE, BODY)
    ax.add_patch(Rectangle((72, 52), 12, 12, fc="#cfcfcf", ec="none", alpha=0.5))
    ax.plot([70, 86], [66, 50], color=S.INK, lw=1.2)
    _lab(ax, 64, 40, "another patient's day:\nnever compared", PT_FINE, SECOND)
    _lab(ax, 0, 14, "Only comparisons inside a patient count, so differences\nin how often "
                    "patients attack cannot help.", PT_FINE, THIRD)


def _how_brier(ax, hea):
    """Brier compares the squared miss of the forecast with that of the patient's rate."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    _lab(ax, 0, 95, "how it is counted", PT_BODY, BODY, weight="bold")
    ax.plot([8, 92], [72, 72], color=S.GREY, lw=1.0)
    for x, t in ((8, "0"), (92, "1")):
        ax.plot([x, x], [70, 74], color=S.GREY, lw=1.0)
        _lab(ax, x, 65, t, PT_FINE, SECOND, ha="center")
    ax.plot([92], [72], "o", ms=6, mfc=S.INK, mec="none")
    _lab(ax, 92, 80, "what happened", PT_BODY, BODY, ha="right")
    for x, col, lab, side in ((66, hea, "forecast", 1), (36, "#9a9a9a", "own rate", -1)):
        ax.plot([x], [72], "o", ms=6, mfc="white", mec=col, mew=1.6)
        ax.plot([x, 92], [72, 72], color=col, lw=2.2, alpha=0.55,
                solid_capstyle="butt")
        sq = (92 - x) * 0.42
        ax.add_patch(Rectangle((x, 30), sq, sq, fc=col, ec="none", alpha=0.35))
        _lab(ax, x, 26, lab, PT_BODY, BODY)
    _lab(ax, 0, 10, "Each miss is squared, so the areas compare. Skill is\nhow much smaller "
                    "the model's area is. 0.00 means equal.", PT_FINE, THIRD)


def _how_dca(ax, att):
    """Net benefit trades attacks caught against unnecessary treatment."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    _lab(ax, 0, 95, "how it is counted", PT_BODY, BODY, weight="bold")
    for i in range(6):
        ax.add_patch(Rectangle((10 + i * 8.5, 62), 7, 14, fc=att, ec="none"))
    _lab(ax, 10, 82, "attacks caught", PT_BODY, BODY)
    for i in range(4):
        ax.add_patch(Rectangle((10 + i * 8.5, 36), 7, 14, fc="#cfcfcf", ec="none"))
    _lab(ax, 10, 56, "days treated for nothing", PT_BODY, BODY)
    _lab(ax, 46, 43, "weighted by how\nreluctant to treat", PT_FINE, SECOND)
    ax.plot([8, 92], [26, 26], color=S.GREY, lw=1.0)
    _lab(ax, 0, 14, "Net benefit subtracts the second from the first. A model\nearns a "
                    "decision only above both default policies.", PT_FINE, THIRD)


def _skill_panel(ax, rows, cols):
    ax.axvline(0.0, color=S.INK, lw=1.0, zorder=3)
    for i, ((tgt, r), col) in enumerate(zip(rows, cols)):
        yy = 1 - i
        lo, hi = float(r["bs_cluster_lo"]), float(r["bs_cluster_hi"])
        ax.plot([lo, hi], [yy, yy], color=col, lw=2.6, solid_capstyle="round", zorder=4)
        ax.plot([float(r["brier_skill"])], [yy], "o", ms=7, mfc="white", mec=col,
                mew=2.0, zorder=5)
        _lab(ax, hi + 0.02, yy, tgt, PT_BODY, BODY)
    ax.annotate("adds nothing beyond\nthe patient's own rate", xy=(0.0, 1.62),
            xytext=(0.075, 1.70), fontsize=PT_FINE, color=SECOND,
            va="center", ha="left",
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
    _lab(ax, 0.20, -0.041, "headache, reported: above treat-none, below treat-everyone",
         PT_FINE, SECOND)
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
    ax.add_patch(Rectangle((0, 30), 4.6, 46, fc=S.INK, ec="none"))
    ax.text(1.1, 53, str(n), fontsize=PT_BODY, fontweight="bold", color="white",
            ha="center", va="center")
    ax.text(6.5, 62, question, fontsize=PT_HEAD, fontweight="bold", color=S.INK,
            ha="left", va="center")
    ax.text(6.5, 33, gloss, fontsize=PT_BODY, color=SECOND, ha="left", va="center")


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

    fig = plt.figure(figsize=(183 / 25.4, 246 / 25.4))
    gs = fig.add_gridspec(9, 2, width_ratios=[0.80, 1.0],
                          height_ratios=[0.30, 1.0, 0.30, 1.0, 0.30, 1.0, 0.30, 1.0, 0.16],
                          left=0.055, right=0.985, top=0.972, bottom=0.018,
                          wspace=0.55, hspace=0.34)

    _question(fig, gs, 0, 1, "Can it rank attack days across the cohort?",
              "Pooled AUROC: every patient's days placed in one ranking.")
    _how_auroc(fig.add_subplot(gs[1, 0]), ORA, BLU)
    GA._draw_slopegraph(fig.add_subplot(gs[1, 1]))

    _question(fig, gs, 2, 2, "Can it rank days inside one patient?",
              "Within-person C-statistic: the same test, run inside each diary.")
    _how_within(fig.add_subplot(gs[3, 0]), ORA)
    GA._draw_per_patient(fig.add_subplot(gs[3, 1]))

    _question(fig, gs, 4, 3, "Does it beat the patient's own attack rate?",
              "Brier skill against each patient's recorded rate.")
    _how_brier(fig.add_subplot(gs[5, 0]), BLU)
    _skill_panel(fig.add_subplot(gs[5, 1]),
                 [("migraine", mig_row), ("headache", hea_row)], [ORA, BLU])

    _question(fig, gs, 6, 4, "Would acting on it help the patient?",
              "Decision-curve net benefit against the two default policies.")
    _how_dca(fig.add_subplot(gs[7, 0]), ORA)
    _dca_panel(fig.add_subplot(gs[7, 1]), preds, ORA)

    axf = fig.add_subplot(gs[8, :]); axf.axis("off")
    axf.set_xlim(0, 100); axf.set_ylim(0, 100)
    axf.text(0, 60, "A reader who stops at question 1 sees a usable forecast. "
                    "No target reaches question 4.", fontsize=8.2, fontweight="bold",
             color=S.INK, va="center")
    axf.text(0, 12, "Headline cells: migraine XGB-HP020, headache TabPFN-v2.6, "
                    "full_features 70/30 chronological. Left column is schematic; "
                    "right column is measured.", fontsize=PT_FINE, color=THIRD, va="center")

    print("saved", S.save(fig, HERE / "figures" / "fig_a6_pooled_vs_within"))


if __name__ == "__main__":
    main()
