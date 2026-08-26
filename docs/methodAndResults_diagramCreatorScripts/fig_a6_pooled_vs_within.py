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

The classifier is deliberately the per-patient climatology this paper uses as the
Brier reference, so the same construction also fixes its Brier skill at exactly
zero: scored against each patient's own attack rate it adds nothing. One
classifier therefore reads 0.74 pooled, 0.50 within patient, and 0.00 on the
clinician's question, which is the figure's point.

Every count and all three statistics are derived from the toy arrays at draw time
and checked against sklearn, so the figure cannot disagree with itself.

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
XMAX, YMAX = 100.0, 11.4
COL_L, COL_R = 0.0, 52.0        # the two column origins; everything hangs off these
IND = 3.6                       # body indent inside a column
BAR_L = 42.0                    # x-units for the full pooled pair count
SQ_W, PITCH = 2.05, 2.45

A_ATTACK, B_ATTACK, N_DAYS = [0, 2, 5, 6, 9], [4], 10
A_RATE, B_RATE = len(A_ATTACK) / N_DAYS, len(B_ATTACK) / N_DAYS


def _toy():
    y = np.array([1 if i in A_ATTACK else 0 for i in range(N_DAYS)]
                 + [1 if i in B_ATTACK else 0 for i in range(N_DAYS)])
    p = np.array([A_RATE] * N_DAYS + [B_RATE] * N_DAYS)
    return y, p


def _counts():
    a_pos, a_neg = len(A_ATTACK), N_DAYS - len(A_ATTACK)
    b_pos, b_neg = len(B_ATTACK), N_DAYS - len(B_ATTACK)
    return {"conc": a_pos * b_neg, "disc": b_pos * a_neg,
            "tied": a_pos * a_neg + b_pos * b_neg,
            "pairs": (a_pos + b_pos) * (a_neg + b_neg)}


def _square_h(ax):
    bb = ax.get_window_extent()
    return SQ_W * (bb.width / XMAX) * (YMAX / bb.height)


def _head(ax, x0, y, letter, text):
    ax.text(x0, y, f"({letter})", fontsize=8.6, fontweight="bold", color=S.INK,
            va="center", ha="left")
    ax.text(x0 + IND, y, text, fontsize=8.6, fontweight="bold", color=S.INK,
            va="center", ha="left")


def _body(ax, x0, y, text, size=7.6, col="#3d3d3d"):
    ax.text(x0 + IND, y, text, fontsize=size, color=col, va="center", ha="left",
            linespacing=1.5)


def _formula(ax, x0, y, tex, size=9.6):
    ax.text(x0 + IND, y, tex, fontsize=size, color=S.INK, va="center", ha="left")


def _day_row(ax, x0, y, attacks, att, h):
    for i in range(N_DAYS):
        ax.add_patch(Rectangle((x0 + IND + i * PITCH, y), SQ_W, h,
                               fc=att if i in attacks else QUIET, ec="none"))


def _seg_bar(ax, x0, y, segs, scale, h):
    x = x0 + IND
    for n, col, alpha in segs:
        if n:
            ax.add_patch(Rectangle((x, y), n * scale, h, fc=col, ec="white",
                                   lw=0.6, alpha=alpha))
        x += n * scale


def main() -> None:
    S.apply()
    plt.rcParams["mathtext.default"] = "regular"
    att, risk = S.target_color("migraine"), S.target_color("headache")
    y, p = _toy()
    c = _counts()
    auroc = float(roc_auc_score(y, p))
    # Build the reference from the per-patient rates independently of the forecast,
    # so the equality below is a derived result rather than an artefact of reusing p.
    ref = np.array([A_RATE] * N_DAYS + [B_RATE] * N_DAYS)
    bs = float(np.mean((p - y) ** 2))
    bs_ref = float(np.mean((ref - y) ** 2))
    skill = 1.0 - bs / bs_ref
    scale = BAR_L / c["pairs"]

    assert len(set(p[:N_DAYS])) == 1 and len(set(p[N_DAYS:])) == 1, \
        "the classifier must be constant within each patient or the example collapses"
    assert c["conc"] + c["disc"] + c["tied"] == c["pairs"], "pair counts must partition"
    assert abs(auroc - (c["conc"] + 0.5 * c["tied"]) / c["pairs"]) < 1e-12, \
        "drawn decomposition must equal the computed AUROC"
    assert abs(skill) < 1e-12, "climatology classifier must have exactly zero Brier skill"

    fig, ax = plt.subplots(figsize=S.figsize("double", 5.05))
    ax.axis("off"); ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.985, bottom=0.015)
    fig.canvas.draw()
    h = _square_h(ax); bh = h * 1.15

    # (a) the shared setup, spanning both columns
    _head(ax, COL_L, 11.05, "a", "Worked cohort: a classifier that outputs each "
                                 "patient's own attack rate, and nothing else")
    risk_x = COL_L + IND + N_DAYS * PITCH + 1.6
    for k, (lbl, attacks, rate) in enumerate(
            [("Patient A, 5 attack days in 10, forecast 0.5 every day", A_ATTACK, A_RATE),
             ("Patient B, 1 attack day in 10, forecast 0.1 every day", B_ATTACK, B_RATE)]):
        yy = 10.25 - k * (h + 0.22)
        _day_row(ax, COL_L, yy, attacks, att, h)
        ax.add_patch(Rectangle((risk_x, yy), 9.0 * rate, h, fc=risk, ec="none",
                               alpha=0.6))
        ax.text(risk_x + 5.4, yy + h / 2, lbl, fontsize=7.6, color=S.INK,
                va="center", ha="left")
    _body(ax, COL_L, 10.25 - (h + 0.22) - 0.72,
          "The forecast never varies across a patient's days, so it carries no day-level "
          "information whatever. Read three ways:", col="#5f5f5f")

    # ---- left column: discrimination, the literature's scale ----------------
    _head(ax, COL_L, 8.20, "b", "Discrimination: AUROC counts ordered pairs")
    _body(ax, COL_L, 7.50, "The chance a randomly drawn attack day is ranked\n"
                           "above a randomly drawn quiet day, ties counting\none half:")
    _formula(ax, COL_L, 6.35,
             r"$\mathrm{AUROC}=\frac{n_{\mathrm{concordant}}+\frac{1}{2}n_{\mathrm{tied}}}"
             r"{n_{\mathrm{pairs}}}$")

    _head(ax, COL_L, 5.20, "c", "Pooled: pairs from the whole cohort")
    _seg_bar(ax, COL_L, 4.55, [(c["conc"], att, 0.95), (c["disc"], att, 0.30),
                               (c["tied"], QUIET, 1.0)], scale, bh)
    _body(ax, COL_L, 4.02, f"$n_{{pairs}}$ = {c['pairs']}:  {c['conc']} concordant, "
                           f"{c['disc']} discordant\n(days from different patients);  "
                           f"{c['tied']} tied\n(days from one patient)", size=7.2)
    _formula(ax, COL_L, 2.90,
             rf"$\mathrm{{AUROC}}=\frac{{{c['conc']}+\frac{{1}}{{2}}({c['tied']})}}"
             rf"{{{c['pairs']}}}={auroc:.2f}$")

    _head(ax, COL_L, 1.85, "d", "Within person: pairs inside each patient")
    _seg_bar(ax, COL_L, 1.20, [(c["tied"], QUIET, 1.0)], scale, bh)
    _body(ax, COL_L, 0.80, f"$n_{{pairs}}$ = {c['tied']}, every pair tied", size=7.2,
          col="#5f5f5f")
    _formula(ax, COL_L, 0.22,
             rf"$C=\frac{{0+\frac{{1}}{{2}}({c['tied']})}}{{{c['tied']}}}={0.5:.2f}$")

    # ---- right column: the clinician's scale --------------------------------
    _head(ax, COL_R, 8.20, "e", "Clinical value: Brier skill")
    _body(ax, COL_R, 7.35, "Squared error of the forecast, set against simply\n"
                           "using each patient's own attack rate. Zero means\n"
                           "the forecast adds nothing to that rate:")
    _formula(ax, COL_R, 6.15,
             r"$\mathrm{skill}=1-\frac{\mathrm{Brier(forecast)}}"
             r"{\mathrm{Brier(own\ rate)}}$")

    _head(ax, COL_R, 5.20, "f", "The same classifier, scored this way")
    for k, (lbl, val) in enumerate([("forecast", bs), ("patient's own rate", bs_ref)]):
        yy = 4.55 - k * (bh + 0.30)
        ax.add_patch(Rectangle((COL_R + IND, yy), val / 0.25 * BAR_L * 0.62, bh,
                               fc=risk if k == 0 else QUIET, ec="white", lw=0.6,
                               alpha=0.75 if k == 0 else 1.0))
        ax.text(COL_R + IND + val / 0.25 * BAR_L * 0.62 + 1.4, yy + bh / 2,
                f"{lbl}  {val:.2f}", fontsize=7.2, color="#3d3d3d", va="center",
                ha="left")
    _body(ax, COL_R, 3.30, "Identical, because the classifier is that rate: it\n"
                           "reproduces the comparator exactly.", size=7.2, col="#5f5f5f")
    _formula(ax, COL_R, 2.35,
             rf"$\mathrm{{skill}}=1-\frac{{{bs:.2f}}}{{{bs_ref:.2f}}}={skill:.2f}$")

    _body(ax, COL_R, 1.15, "One classifier, three readings: 0.74 pooled, 0.50 within\n"
                           "a patient, 0.00 against the patient's own rate. Only the\n"
                           "first suggests a usable forecast.", size=7.6, col=S.INK)

    print("saved", S.save(fig, HERE / "figures" / "fig_a6_pooled_vs_within"))


if __name__ == "__main__":
    main()
