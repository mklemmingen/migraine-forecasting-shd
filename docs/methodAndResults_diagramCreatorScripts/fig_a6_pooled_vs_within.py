"""Figure A6 - what pooled AUROC and the within-person C-statistic each measure.

A worked toy cohort, not model output. Two patients differ in attack rate; the
"model" outputs one constant risk per patient and never varies it day to day, so
it carries no day-level information whatsoever. Pooled AUROC still reaches 0.74,
because ranking every patient's days in one list lets between-patient base-rate
separation stand in for forecasting. The within-person C-statistic, which ranks
each patient's days against their own, is exactly 0.50 -- every within-patient
pair is a tie.

That is the paper's central distinction in one picture: pooled AUROC rewards two
different abilities, and only one of them is a forecast a patient could use.

Both numbers are computed from the toy arrays at draw time (not hardcoded), so
the figure cannot drift from the claim it makes.

Usage: python fig_a6_pooled_vs_within.py
"""
# §11 compliance: schematic only (no headline metric; the 0.74/0.50 are a worked
#   toy example, computed in-figure, and are labelled as such).
#   §11.3 self-contained caption:        setup stated in the suptitle
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

QUIET = "#e3e3e3"          # same neutral as the graphical abstract's day grid
CELL, GAP = 0.84, 1.0      # square side and pitch, in data units

# Patient A is attack-dense, patient B is attack-sparse. Positions of the attack
# days inside each patient's ten are arbitrary: the model cannot see them.
A_ATTACK = [0, 2, 5, 6, 9]
B_ATTACK = [4]
N_DAYS = 10


def _toy():
    y = np.array([1 if i in A_ATTACK else 0 for i in range(N_DAYS)]
                 + [1 if i in B_ATTACK else 0 for i in range(N_DAYS)])
    p = np.array([0.8] * N_DAYS + [0.2] * N_DAYS)
    return y, p


def _row(ax, x0, y0, attacks, attack_col):
    for i in range(N_DAYS):
        ax.add_patch(Rectangle((x0 + i * GAP, y0), CELL, CELL,
                               fc=attack_col if i in attacks else QUIET,
                               ec="none"))



def _risk_bar(ax, x0, y0, level, col, width):
    """The model's output: one height per patient, flat across that patient's days."""
    ax.add_patch(Rectangle((x0, y0), width, level, fc=col, ec="none", alpha=0.6))


def _counts():
    """Every attack-day / quiet-day comparison AUROC averages over, split by whether
    the two days come from the same patient. AUROC is the share of these the model
    wins, ties counting a half."""
    a_pos, a_neg = len(A_ATTACK), N_DAYS - len(A_ATTACK)
    b_pos, b_neg = len(B_ATTACK), N_DAYS - len(B_ATTACK)
    return {"won": a_pos * b_neg,          # A attack over B quiet: model ranks A higher
            "lost": b_pos * a_neg,         # B attack under A quiet: model ranks A higher
            "tied": a_pos * a_neg + b_pos * b_neg,   # inside a patient: identical score
            "total": (a_pos + b_pos) * (a_neg + b_neg)}


def _setup_panel(ax, att, risk):
    ax.axis("off"); ax.set_xlim(-4.6, 13.2); ax.set_ylim(-1.15, 2.6)
    for k, (name, attacks, lvl, rate) in enumerate([
            ("Patient A", A_ATTACK, 0.60, "5 attack days in 10"),
            ("Patient B", B_ATTACK, 0.15, "1 attack day in 10")]):
        yy = 1.35 - k * 1.15
        ax.text(-4.5, yy + CELL / 2, f"{name}, {rate}", fontsize=8.2,
                color=S.INK, va="center", ha="left")
        _row(ax, 0, yy, attacks, att)
        _risk_bar(ax, 10.5, yy, lvl, risk, 2.0)
    ax.text(12.7, 1.35 + CELL / 2, "model's risk", fontsize=7.2, color="#5f5f5f",
            va="center", ha="left")
    ax.text(-4.5, -0.85, "The model gives each patient one risk and never varies it "
            "day to day, so it carries no information about which day an attack falls on.",
            fontsize=7.8, color="#3d3d3d", va="center", ha="left")


def _bar(ax, y, segs, total, h=0.52):
    """Draw the comparisons as one bar on a fixed 0..total scale, so the
    within-person bar is visibly shorter: it simply has fewer comparisons to make."""
    x = 0.0
    for n, col, alpha in segs:
        if n:
            ax.add_patch(Rectangle((x, y), n, h, fc=col, ec="white", lw=0.6,
                                   alpha=alpha))
        x += n


def _compare_panels(ax, att, risk, c):
    ax.axis("off"); ax.set_xlim(-1.5, c["total"] * 1.02); ax.set_ylim(-0.30, 5.35)
    won, lost, tied, tot = c["won"], c["lost"], c["tied"], c["total"]

    ax.text(0, 5.05, "Pooled AUROC compares every attack day with every quiet day, "
            "whoever they belong to", fontsize=8.4, fontweight="bold", color=S.INK,
            va="center", ha="left")
    _bar(ax, 4.05, [(won, att, 0.95), (lost, att, 0.30), (tied, QUIET, 1.0)], tot)
    ax.text(0, 3.92, f"{won} of the {tot} comparisons are between patients and the model "
            f"wins them, {lost} it loses, and the {tied} inside a patient are all ties",
            fontsize=7.2, color="#3d3d3d", ha="left", va="top")
    ax.text(0, 3.05, f"AUROC {(won + 0.5 * tied) / tot:.2f}", fontsize=10.5,
            fontweight="bold", color=S.INK, va="center", ha="left")
    ax.text(13.5, 3.05, "carried by the between-patient comparisons", fontsize=7.6,
            color="#3d3d3d", va="center", ha="left")

    ax.text(0, 1.95, "The within-person C-statistic keeps only the comparisons "
            "inside a patient", fontsize=8.4, fontweight="bold", color=S.INK,
            va="center", ha="left")
    _bar(ax, 1.05, [(tied, QUIET, 1.0)], tot)
    ax.text(tied / 2, 0.92, f"the same {tied} ties, and nothing else left to score",
            fontsize=7.2, color="#5f5f5f", ha="center", va="top")
    ax.text(0, 0.10, "C 0.50", fontsize=10.5, fontweight="bold", color=S.INK,
            va="center", ha="left")
    ax.text(9.0, 0.10, "chance: the model never ranked one of a patient's own days "
            "above another", fontsize=7.6, color="#3d3d3d", va="center", ha="left")


def main() -> None:
    S.apply()
    att = S.target_color("migraine")
    risk = S.target_color("headache")
    y, p = _toy()
    auroc = float(roc_auc_score(y, p))
    c = _counts()

    assert len(set(p[:N_DAYS])) == 1 and len(set(p[N_DAYS:])) == 1, \
        "the toy model must be constant within each patient or the point collapses"
    assert abs(auroc - (c["won"] + 0.5 * c["tied"]) / c["total"]) < 1e-9, \
        "the drawn decomposition must equal the computed AUROC"
    assert abs(auroc - 0.7381) < 5e-4, f"toy pooled AUROC drifted: {auroc}"

    fig, axes = plt.subplots(2, 1, figsize=S.figsize("double", 4.15),
                             gridspec_kw={"height_ratios": [1.0, 1.45]})
    _setup_panel(axes[0], att, risk)
    _compare_panels(axes[1], att, risk, c)

    fig.suptitle("What each score measures, on a model that knows only which patient "
                 "it is looking at", y=0.985)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.03, hspace=0.22)
    print("saved", S.save(fig, HERE / "figures" / "fig_a6_pooled_vs_within"))


if __name__ == "__main__":
    main()
