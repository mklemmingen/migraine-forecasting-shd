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

Usage: python fig_m_evaluation_series.py
       The decision-curve panel reads figures/headline_predictions.npz; rebuild that
       cache with make_headline_predictions.py in this directory.
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

# Panel (a) is achromatic. The blind readability study found five independent readers
# reading panel (a)'s hues as a colour key for panel (b) -- orange meant "attack day" on
# the left and "migraine" on the right. A grey has no chroma, so the confusion cannot
# survive: every schematic mark is R=G=B, every data mark carries a target hue.
A_INK, A_STRONG, A_MID = "#222222", "#4a4a4a", "#8e8e8e"
A_LIGHT, A_EDGE, A_PANE = "#d8d8d8", "#9a9a9a", "#f2f2f2"
# target words: #D55E00 is 3.87:1 on white and fails as text; #C25100 is 4.70:1
MIG_TEXT, HEA_TEXT = "#C25100", "#0072B2"


def _lab(ax, x, y, t, size=PT_BODY, col=None, ha="left", weight="normal",
         rot=0):
    ax.text(x, y, t, fontsize=size, color=col or SECOND, ha=ha, va="center",
            fontweight=weight, linespacing=1.5, rotation=rot)


def _sq(ax, x, y, side, col, alpha=1.0, anchor="center", outline=False):
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
    if outline:
        ax.add_patch(Rectangle((x, y0), side, h, fc="none", ec=col, lw=1.0,
                               ls=(0, (2.5, 1.8)), alpha=alpha))
    else:
        ax.add_patch(Rectangle((x, y0), side, h, fc=col, ec="none", alpha=alpha))
    return h


def _how_auroc(ax):
    """AUROC counts ordered pairs: one attack day against one quiet day."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.annotate("", xy=(14, 92), xytext=(14, 26),
                arrowprops=dict(arrowstyle="-|>", color=A_MID, lw=1.0))
    _lab(ax, 6, 60, "forecast risk", PT_FINE, SECOND, ha="center", rot=90)
    for yy, col, lab in ((78, A_STRONG, "attack day"), (44, A_LIGHT, "quiet day")):
        _sq(ax, 26, yy, 11, col)
        _lab(ax, 42, yy, lab, PT_BODY, BODY)
    ax.annotate("", xy=(32, 70), xytext=(32, 52),
                arrowprops=dict(arrowstyle="-|>", color=A_INK, lw=1.2))
    _lab(ax, 44, 61, "ranked\ncorrectly", PT_BODY, BODY, weight="bold")
    # the readers could describe the pair test and still not know AUROC *is* the pair count
    _lab(ax, 6, 14, "AUROC = share of pairs\nranked this way", PT_BODY, BODY)


def _how_within(ax):
    """The same pair test, confined to one patient's own diary."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.add_patch(Rectangle((4, 44), 54, 46, fc=A_PANE, ec="#e4e4e4", lw=0.6))
    _lab(ax, 8, 84, "one patient's diary", PT_FINE, SECOND)
    for yy, col, lab in ((72, A_STRONG, "attack day"), (52, A_LIGHT, "quiet day")):
        _sq(ax, 12, yy, 10, col)
        _lab(ax, 24, yy, lab, PT_BODY, BODY)
    ax.annotate("", xy=(17, 68), xytext=(17, 56),
                arrowprops=dict(arrowstyle="<|-|>", color=A_INK, lw=1.1,
                                mutation_scale=7))
    _lab(ax, 24, 62, "compared", PT_FINE, BODY, weight="bold")
    _sq(ax, 74, 66, 10, A_EDGE, outline=True)
    _lab(ax, 79, 52, "another patient's day,\nnever compared", PT_FINE, SECOND,
         ha="center")
    _lab(ax, 6, 18, "AUROC inside one diary\n= within-person C", PT_BODY, BODY)


def _how_brier(ax):
    """One forecast, both outcomes: the same prediction wins on an attack day and
    loses on a quiet day. The single-outcome version taught "higher is better",
    which is wrong on the ~93% of days when nothing happens."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    OWN, FC = 0.20, 0.45                       # illustrative, stated in the caption
    x0, x1 = 8.0, 66.0
    px = lambda q: x0 + q * (x1 - x0)
    for y, outcome, head, ticks in ((79, 1.0, "attack day", True),
                                    (31, 0.0, "quiet day  (most days)", False)):
        _lab(ax, x0, y + 14, head, PT_BODY, BODY, weight="bold")
        if ticks:                                   # one probability scale, labelled once
            _lab(ax, x0, y + 7, "0", PT_FINE, THIRD, ha="center")
            _lab(ax, x1, y + 7, "1", PT_FINE, THIRD, ha="center")
        ax.plot([x0, x1], [y, y], color=A_MID, lw=1.0, zorder=1)
        for q in (0.0, 1.0):
            ax.plot([px(q), px(q)], [y - 2, y + 2], color=A_MID, lw=1.0)
        ax.plot([px(outcome)], [y], "o", ms=5.5, mfc=A_INK, mec="none", zorder=4)
        for q, col, dy, nm in ((OWN, A_MID, -8.0, "own rate"), (FC, A_INK, -14.5, "forecast")):
            ax.plot([px(q)], [y], "o", ms=5.0, mfc="white", mec=col, mew=1.4, zorder=4)
            w = (q - outcome) ** 2 * (x1 - x0)
            ax.add_patch(Rectangle((x0, y + dy), w, 3.6, fc=col, ec="none", zorder=3))
            if ticks:                               # name the bars once, top row only
                _lab(ax, x0 + w + 3, y + dy + 1.8, nm, PT_FINE, col)
    _lab(ax, x0, 54, "bar length = squared miss,  shorter = better", PT_FINE, THIRD)
    _lab(ax, x0, 5, "skill = 1 \u2212 (forecast \u00f7 own-rate)", PT_BODY, BODY)


def _how_dca(ax):
    """Net benefit trades attacks caught against unnecessary treatment."""
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    for i in range(6):
        _sq(ax, 12 + i * 8.5, 66, 7, A_STRONG, anchor="bottom")
    _lab(ax, 12, 80, "attacks caught", PT_BODY, BODY)
    # the operator, so the panel shows a subtraction rather than two unrelated rows
    ax.text(4, 50, "\u2212", fontsize=13, color=A_INK, ha="center", va="center")
    for i in range(4):
        _sq(ax, 12 + i * 8.5, 26, 7, A_LIGHT, anchor="bottom")
    _lab(ax, 12, 40, "days treated for nothing", PT_BODY, BODY)
    # was "weighted by how reluctant one is to treat" -- a clause, and it never named
    # the quantity. These two fragments name it and tie it to the panel (b) x-axis.
    _lab(ax, 12, 14, "weight = t / (1 \u2212 t)", PT_FINE, SECOND)
    _lab(ax, 12, 6, "false alarms per attack caught", PT_FINE, SECOND)


def _skill_panel(ax, rows, cols):
    ax.axvline(0.0, color=S.INK, lw=1.0, zorder=3)
    for i, ((tgt, r), col) in enumerate(zip(rows, cols)):
        yy = 1 - i
        lo, hi = float(r["bs_cluster_lo"]), float(r["bs_cluster_hi"])
        ax.plot([lo, hi], [yy, yy], color=col, lw=2.6, solid_capstyle="round", zorder=4)
        ax.plot([float(r["brier_skill"])], [yy], "o", ms=7, mfc="white", mec=col,
                mew=2.0, zorder=5)
        _lab(ax, hi + 0.02, yy, tgt.upper(), PT_VALUE, MIG_TEXT if tgt == "migraine" else HEA_TEXT, weight="bold")
    _lab(ax, 0.0, 1.46, "no improvement", PT_FINE, SECOND, ha="center")
    ax.set_ylim(-0.62, 1.62); ax.set_xlim(-0.32, 0.72)  # room for the 11pt target word
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
    # treat-all is a POLICY, not a target, so it is achromatic. Both blind readers read
    # the old same-hue dashed line as a confidence band on the model.
    ax.plot(ts, nb_all, color=A_MID, lw=0.9, ls=(0, (4, 2)), zorder=4)
    ax.plot(ts, nb, color=col, lw=2.0, zorder=5, solid_capstyle="round")
    j = int(len(ts) * 0.34)
    _lab(ax, ts[j], nb[j] + 0.010, "MIGRAINE", PT_VALUE, MIG_TEXT, weight="bold")
    _lab(ax, 0.505, 0.0, "treat none", PT_FINE, BODY)
    _lab(ax, 0.145, -0.041, "treat everyone", PT_FINE, SECOND)
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


def _one(name, how, how_args, draw_right, right_args, after=None):
    """One figure in the series: the schematic beside the quantity it explains.

    Each is full column width and short, the shape this literature uses for a
    full-width explanatory figure. The prose that used to sit under each schematic
    now lives in the manuscript caption, which is where a journal expects it.
    """
    fig = plt.figure(figsize=(183 / 25.4, 62 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.46, 1.0],
                          left=0.055, right=0.985, top=0.88, bottom=0.185,
                          wspace=0.30)
    axl, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axl.set_xlim(0, 100); axl.set_ylim(0, 100); axl.axis("off")
    fig.canvas.draw()
    _panel_letter(axl, "a"); _panel_letter(axr, "b")
    how(axl, *how_args)
    draw_right(axr, *right_args)
    if after is not None:
        after(axr)          # annotate on top; the GA's own drawing is never modified
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

    def _label_slope(ax):
        """The blind study: neither m1 reader ever learned what orange and blue were."""
        ax.text(0.24, 0.815, "MIGRAINE", fontsize=PT_VALUE, fontweight="bold",
                color=MIG_TEXT, ha="left", va="bottom", zorder=6)
        ax.text(0.08, 0.548, "HEADACHE", fontsize=PT_VALUE, fontweight="bold",
                color=HEA_TEXT, ha="left", va="top", zorder=6)

    def _label_rank(ax):
        for x, y, t, c in ((0.030, 0.905, "MIGRAINE", MIG_TEXT),
                           (0.030, 0.835, "HEADACHE", HEA_TEXT)):
            ax.text(x, y, t, fontsize=PT_VALUE, fontweight="bold", color=c,
                    ha="left", va="center", zorder=6)
        # Both blind readers counted the dots and asked whether the cohort was 76 people.
        # It is not: the 19 migraine-scorable records are a strict subset of the 57
        # headache-scorable ones, both drawn from the same 63. Overridden here rather
        # than in graphical_abstract.py, which must keep rendering unchanged.
        ax.set_xlabel("one dot per patient\n"
                      "(63 records; 57 scorable for headache, 19 for migraine)",
                      fontsize=PT_AXIS)

    _one("fig_m1_pooled_auroc", _how_auroc, (), GA._draw_slopegraph, (), _label_slope)
    _one("fig_m2_within_person", _how_within, (), GA._draw_per_patient, (), _label_rank)
    _one("fig_m3_brier_skill", _how_brier, (), _skill_panel,
         ([("migraine", mig_row), ("headache", hea_row)], [ORA, BLU]))
    _one("fig_m4_net_benefit", _how_dca, (), _dca_panel, (preds, ORA))


if __name__ == "__main__":
    main()
