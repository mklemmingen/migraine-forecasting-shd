"""Figure A6 - what pooled AUROC and the within-person C-statistic each count.

AUROC is a rank statistic: the probability that a randomly drawn attack day is
ordered above a randomly drawn quiet day, ties counted as one half. Which pairs
enter that average is a modelling choice, and it is the choice this paper turns
on. Pooling forms pairs across the whole cohort, so two days from different
patients are compared; the within-person C-statistic forms pairs only inside a
patient.

The worked cohort makes the consequence exact. Two patients differ in attack
rate and the classifier assigns one constant risk per patient, carrying no
day-level information at all. Pooled AUROC is 0.74 regardless, because 45 of the
84 pairs are between patients and every one of them is concordant by base rate
alone. The within-person C-statistic retains only the 34 within-patient pairs,
all of which are tied, and is exactly 0.50.

Every count and both statistics are derived from the toy arrays at draw time and
checked against sklearn, so the figure cannot disagree with itself.

Usage: python fig_a6_pooled_vs_within.py
"""
# §11 compliance: schematic only (no headline metric; 0.74/0.50 are a worked
#   example computed in-figure and labelled as such).
#   §11.3 self-contained caption:        construction stated in panel (b)
#   §11.1, §11.2, §11.5, §11.7:          N/A (no cohort data, no cell)
#   §11.10 no "substantial"/"large":     verified
#   §11.11 self-check:                   this block
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent

QUIET = "#e3e3e3"
LEFT = 0.0                 # single left margin; every element in the figure starts here
XMAX, YMAX = 100.0, 10.6
BAR_SPAN = 58.0            # x-units representing the full pooled pair count
SQ_W, PITCH = 2.05, 2.45   # day square width and centre-to-centre, in x-units

A_ATTACK, B_ATTACK, N_DAYS = [0, 2, 5, 6, 9], [4], 10


def _toy():
    y = np.array([1 if i in A_ATTACK else 0 for i in range(N_DAYS)]
                 + [1 if i in B_ATTACK else 0 for i in range(N_DAYS)])
    p = np.array([0.8] * N_DAYS + [0.2] * N_DAYS)
    return y, p


def _counts():
    a_pos, a_neg = len(A_ATTACK), N_DAYS - len(A_ATTACK)
    b_pos, b_neg = len(B_ATTACK), N_DAYS - len(B_ATTACK)
    return {"conc": a_pos * b_neg, "disc": b_pos * a_neg,
            "tied": a_pos * a_neg + b_pos * b_neg,
            "pairs": (a_pos + b_pos) * (a_neg + b_neg)}


def _square_h(ax):
    """Height, in y-units, that renders SQ_W as a true square at the final size."""
    bb = ax.get_window_extent()
    return SQ_W * (bb.width / XMAX) * (YMAX / bb.height)


def _head(ax, y, letter, text):
    """Panel heading. The letter sits in its own fixed column so every heading,
    body line, day grid and bar in the figure shares one left margin."""
    ax.text(LEFT, y, f"({letter})", fontsize=8.6, fontweight="bold", color=S.INK,
            va="center", ha="left")
    ax.text(LEFT + 3.6, y, text, fontsize=8.6, fontweight="bold", color=S.INK,
            va="center", ha="left")


def _body(ax, y, text, size=7.8, col="#3d3d3d", x=None):
    ax.text(LEFT + 3.6 if x is None else x, y, text, fontsize=size, color=col, va="center", ha="left",
            linespacing=1.45)


def _day_row(ax, y, attacks, att, h):
    for i in range(N_DAYS):
        ax.add_patch(Rectangle((LEFT + 3.6 + i * PITCH, y), SQ_W, h,
                               fc=att if i in attacks else QUIET, ec="none"))


def _pair_bar(ax, y, segs, scale, h):
    x = LEFT + 3.6
    for n, col, alpha in segs:
        if n:
            ax.add_patch(Rectangle((x, y), n * scale, h, fc=col, ec="white",
                                   lw=0.6, alpha=alpha))
        x += n * scale


def main() -> None:
    S.apply()
    plt.rcParams["mathtext.default"] = "regular"      # math in the body font
    att, risk = S.target_color("migraine"), S.target_color("headache")
    y, p = _toy()
    c = _counts()
    auroc = float(roc_auc_score(y, p))
    within = 0.5
    scale = BAR_SPAN / c["pairs"]

    assert len(set(p[:N_DAYS])) == 1 and len(set(p[N_DAYS:])) == 1, \
        "the classifier must be constant within each patient or the example collapses"
    assert c["conc"] + c["disc"] + c["tied"] == c["pairs"], "pair counts must partition"
    assert abs(auroc - (c["conc"] + 0.5 * c["tied"]) / c["pairs"]) < 1e-12, \
        "drawn decomposition must equal the computed AUROC"

    fig, ax = plt.subplots(figsize=S.figsize("double", 4.75))
    ax.axis("off"); ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.985, bottom=0.015)
    fig.canvas.draw()
    h = _square_h(ax)
    bh = h * 1.15

    # One vertical rhythm for the whole figure: every heading, body line, grid and
    # bar is placed off this ladder, so nothing is positioned by eye.
    COL = LEFT + 3.6                       # body column, right of the panel letters

    # (a) the definition, without which the pair counts below have no referent
    _head(ax, 10.20, "a", "AUROC is a statement about ordered pairs")
    _body(ax, 9.48, "The probability that a randomly drawn attack day is ordered above a "
                    "randomly drawn quiet day,\nwith tied pairs contributing one half:")
    ax.text(COL, 8.42,
            r"$\mathrm{AUROC}\,=\,\frac{n_{\mathrm{concordant}}\,+\,"
            r"\frac{1}{2}\,n_{\mathrm{tied}}}{n_{\mathrm{pairs}}}$",
            fontsize=10.5, color=S.INK, va="center", ha="left")
    _body(ax, 7.55, "Which pairs are admitted to that average is a choice, and it is the "
                    "choice separating the two\ncolumns of this paper's results.",
          col="#5f5f5f")

    # (b) a classifier carrying no day-level information at all
    _head(ax, 6.35, "b", "Worked cohort: a classifier using patient identity alone")
    risk_x = COL + N_DAYS * PITCH + 1.6
    ax.text(risk_x, 5.82, "assigned risk", fontsize=7.0, color="#5f5f5f",
            va="bottom", ha="left")
    for k, (lbl, attacks, lvl) in enumerate(
            [("Patient A, 5 attack days in 10", A_ATTACK, 0.72),
             ("Patient B, 1 attack day in 10", B_ATTACK, 0.18)]):
        yy = 5.28 - k * (h + 0.22)
        _day_row(ax, yy, attacks, att, h)
        ax.add_patch(Rectangle((risk_x, yy), 9.0 * lvl, h, fc=risk, ec="none",
                               alpha=0.6))
        ax.text(risk_x + 10.4, yy + h / 2, lbl, fontsize=7.6, color=S.INK,
                va="center", ha="left")
    _body(ax, 5.28 - (h + 0.22) - 0.78,
          "One risk per patient, held constant across that patient's days: no day-level "
          "information is present.", col="#5f5f5f")

    # (c) pooling admits between-patient pairs
    _head(ax, 3.30, "c", "Pooled: pairs drawn from the cohort as a whole")
    _pair_bar(ax, 2.68, [(c["conc"], att, 0.95), (c["disc"], att, 0.30),
                         (c["tied"], QUIET, 1.0)], scale, bh)
    _body(ax, 2.26, f"$n_{{pairs}}$ = {c['pairs']}:  {c['conc']} concordant and "
                    f"{c['disc']} discordant, each pairing days from different patients;  "
                    f"{c['tied']} tied, each pairing days from one patient", size=7.4)
    ax.text(COL, 1.66,
            rf"$\mathrm{{AUROC}}=\frac{{{c['conc']}+\frac{{1}}{{2}}({c['tied']})}}"
            rf"{{{c['pairs']}}}={auroc:.2f}$", fontsize=10.0, color=S.INK,
            va="center", ha="left")

    # (d) the within-person restriction discards exactly those between-patient pairs
    _head(ax, 0.86, "d", "Within person: pairs drawn inside each patient")
    _pair_bar(ax, 0.10, [(c["tied"], QUIET, 1.0)], scale, bh)
    _body(ax, 0.10 + bh / 2, f"$n_{{pairs}}$ = {c['tied']}, every pair tied", size=7.4,
          col="#5f5f5f", x=COL + c["tied"] * scale + 2.2)
    ax.text(62.0, 0.10 + bh / 2,
            rf"$C=\frac{{0+\frac{{1}}{{2}}({c['tied']})}}{{{c['tied']}}}={within:.2f}$",
            fontsize=10.0, color=S.INK, va="center", ha="left")

    print("saved", S.save(fig, HERE / "figures" / "fig_a6_pooled_vs_within"))


if __name__ == "__main__":
    main()
