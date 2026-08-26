"""Figure A6 - the two comparisons pooled AUROC and the within-person C ask.

AUROC is an average over attack-day/quiet-day comparisons, so what it measures
depends entirely on which comparisons are allowed into that average. Pooling lets a
day be compared with any other patient's day; the within-person C-statistic allows
only comparisons inside one patient. This figure shows one comparison of each kind,
with the classifier's actual answer, and lets the reader see that the first is easy
for a reason that has nothing to do with forecasting.

The worked classifier emits each patient's own attack rate and nothing else, so it
holds no day-level information. It answers the between-patient comparison correctly
every time (patient A simply has attacks more often) and cannot answer the
within-patient comparison at all (it returns the same number on both days). The same
construction makes it the per-patient climatology this paper uses as the Brier
reference, so its Brier skill is exactly zero.

One classifier, three readings: 0.74 pooled, 0.50 within patient, 0.00 against the
patient's own rate. Every number is derived from the toy arrays at draw time and
checked against sklearn.

Usage: python fig_a6_pooled_vs_within.py
"""
# §11 compliance: schematic only (0.74/0.50/0.00 are a worked example computed
#   in-figure and labelled as such).
#   §11.3 self-contained caption:        construction stated in panel (a)
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
from matplotlib.patches import Rectangle, FancyBboxPatch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent

W_MM, H_MM = 183.0, 150.0          # canvas is millimetres at final size
QUIET, RULE = "#e3e3e3", "#9a9a9a"

N_DAYS = 10
A_ATTACK, B_ATTACK = [0, 2, 5, 6, 9], [4]
A_RATE, B_RATE = len(A_ATTACK) / N_DAYS, len(B_ATTACK) / N_DAYS

# the three days the two questions are built from (0-based indices)
A_ATK_DAY, A_QUIET_DAY, B_QUIET_DAY = 2, 7, 5


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


def _head(ax, x, y, letter, text, size=8.6):
    ax.text(x, y, f"({letter})", fontsize=size, fontweight="bold", color=S.INK,
            va="center", ha="left")
    ax.text(x + 5.6, y, text, fontsize=size, fontweight="bold", color=S.INK,
            va="center", ha="left")


def _note(ax, x, y, text, size=7.2, col="#3d3d3d", ha="left", weight="normal"):
    ax.text(x, y, text, fontsize=size, color=col, va="center", ha=ha,
            linespacing=1.5, fontweight=weight)


def _claim(ax, x, y, letter, text):
    """Panel heading as a sentence stating that panel's claim, after the convention
    this literature uses for teaching figures (e.g. Sebastianelli 2024, Cephalalgia)."""
    ax.text(x, y, f"({letter})", fontsize=8.6, fontweight="bold", color=S.INK,
            va="center", ha="left")
    ax.text(x + 5.6, y, text, fontsize=8.6, fontweight="bold", color=S.INK,
            va="center", ha="left")


def _daycard(ax, x, y, w, h, who, day, is_attack, risk, att):
    """One diary day, drawn big enough to read: whose it is, what happened, and the
    single number the forecast gave it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                fc="white", ec=RULE, lw=0.8))
    _note(ax, x + w / 2, y + h - 4.0, f"{who}, day {day}", size=7.0, col=S.INK,
          ha="center", weight="bold")
    sq = 7.0
    ax.add_patch(Rectangle((x + w / 2 - sq / 2, y + h - 14.5), sq, sq,
                           fc=att if is_attack else QUIET, ec="none"))
    _note(ax, x + w / 2, y + h - 18.6, "attack day" if is_attack else "quiet day",
          size=6.8, col="#5f5f5f", ha="center")
    _note(ax, x + w / 2, y + 4.2, f"forecast {risk:.2f}", size=8.6, col=S.INK,
          ha="center", weight="bold")


def _diaries(ax, att):
    _claim(ax, 4, 145.0, "a", "The forecast is one number per patient, the same on every day")
    s, pitch = 5.2, 6.1
    ring = {("A", A_ATK_DAY), ("A", A_QUIET_DAY), ("B", B_QUIET_DAY)}
    for k, (name, attacks, rate) in enumerate([("A", A_ATTACK, A_RATE),
                                               ("B", B_ATTACK, B_RATE)]):
        yy = 134.0 - k * 8.4
        _note(ax, 4, yy + s / 2, f"Patient {name}", size=7.4, col=S.INK, weight="bold")
        for i in range(N_DAYS):
            ax.add_patch(Rectangle((22 + i * pitch, yy), s, s,
                                   fc=att if i in attacks else QUIET, ec="none"))
            if (name, i) in ring:
                ax.add_patch(FancyBboxPatch((22 + i * pitch - 0.7, yy - 0.7),
                                           s + 1.4, s + 1.4, fc="none", ec=S.INK,
                                           lw=1.1, zorder=5,
                                           boxstyle="round,pad=0,rounding_size=1.2"))
        _note(ax, 22 + N_DAYS * pitch + 3.0, yy + s / 2,
              f"every day forecast {rate:.2f}", size=7.0, col="#5f5f5f")
    _note(ax, 4, 119.0, "Patient A has attacks on half her days, patient B on one day in ten. "
                       "The forecast reports exactly that, and nothing about which day.",
          size=7.2, col="#5f5f5f")
    _note(ax, 22, 112.5, "The three outlined days are the ones compared below.", size=7.0,
          col="#5f5f5f")


def _question(ax, x0, y0, claim, letter, left, right, verdict, ok, why, tally, score,
              score_lab, att):
    """One comparison, drawn at a size a reader can take in: the two days, the two
    numbers, and what the forecast concluded from them."""
    _claim(ax, x0, y0, letter, claim)
    cw, ch, gap = 30.0, 26.0, 13.0
    cy = y0 - 33.0
    _daycard(ax, x0, cy, cw, ch, *left, att)
    _daycard(ax, x0 + cw + gap, cy, cw, ch, *right, att)
    _note(ax, x0 + cw + gap / 2, cy + ch / 2, "vs", size=8.6, col="#5f5f5f", ha="center")

    # the verdict, drawn between the two cards
    vy = cy - 6.0
    ax.plot([x0 + cw / 2, x0 + cw / 2, x0 + cw + gap + cw / 2, x0 + cw + gap + cw / 2],
            [cy - 1.0, vy, vy, cy - 1.0], color=RULE, lw=0.8)
    _note(ax, x0 + cw + gap / 2, vy - 5.0, verdict, size=9.0, col=S.INK, ha="center",
          weight="bold")
    _note(ax, x0 + cw + gap / 2, vy - 10.6,
          "the forecast ranks them correctly" if ok else "the forecast cannot separate them",
          size=7.2, col=S.INK if ok else "#5f5f5f", ha="center")
    _note(ax, x0, vy - 17.5, why, size=7.2, col="#5f5f5f")
    _note(ax, x0, vy - 25.5, tally, size=7.2, col="#3d3d3d")
    _note(ax, x0, vy - 32.5, f"{score_lab} {score:.2f}", size=11.5, col=S.INK,
          weight="bold")


def _brier(ax, bs, bs_ref, skill):
    """Brier skill is not a comparison between two days, so it gets the question a
    clinician would actually put to a forecast: does it beat what we already knew?"""
    _claim(ax, 4, 24.0, "d", "Brier skill asks whether the forecast beats what we "
                             "already knew about that patient")
    bw, bh, gap = 52.0, 12.0, 16.0
    by = 6.5
    for k, (lab, val) in enumerate([("what the forecast says for patient A", "0.50 every day"),
                                    ("what her own attack rate already said", "0.50")]):
        x = 22.0 + k * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, by), bw, bh,
                                    boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="white", ec=RULE, lw=0.8))
        _note(ax, x + bw / 2, by + bh - 4.2, lab, size=6.8, col="#5f5f5f", ha="center")
        _note(ax, x + bw / 2, by + 4.0, val, size=8.4, col=S.INK, ha="center",
              weight="bold")
    _note(ax, 22.0 + bw + gap / 2, by + bh / 2, "=", size=11.0, col=S.INK, ha="center",
          weight="bold")
    _note(ax, 22.0 + 2 * bw + gap + 5.0, by + bh / 2,
          f"skill {skill:.2f}\nthe forecast adds nothing", size=8.0, col=S.INK,
          weight="bold")


def main() -> None:
    S.apply()
    att = S.target_color("migraine")
    y, p = _toy()
    c = _counts()
    auroc = float(roc_auc_score(y, p))
    ref = np.array([A_RATE] * N_DAYS + [B_RATE] * N_DAYS)
    bs, bs_ref = float(np.mean((p - y) ** 2)), float(np.mean((ref - y) ** 2))
    skill = 1.0 - bs / bs_ref

    assert c["conc"] + c["disc"] + c["tied"] == c["pairs"] == 84, "pairs must partition"
    assert abs(auroc - (c["conc"] + 0.5 * c["tied"]) / c["pairs"]) < 1e-12, \
        "the drawn tally must equal the computed AUROC"
    assert abs(skill) < 1e-12, "climatology classifier must have exactly zero Brier skill"
    assert A_ATK_DAY in A_ATTACK and A_QUIET_DAY not in A_ATTACK, "panel b/c days mislabelled"
    assert B_QUIET_DAY not in B_ATTACK, "panel b comparison day mislabelled"

    fig, ax = plt.subplots(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM)
    ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    _diaries(ax, att)
    _question(ax, 4, 103.0,
              "Pooled AUROC mostly compares different patients", "b",
              ("Patient A", A_ATK_DAY + 1, True, A_RATE),
              ("Patient B", B_QUIET_DAY + 1, False, B_RATE),
              "0.50  >  0.10", True,
              "Easy, and for the wrong reason: patient A simply has\nattacks more often "
              "than patient B. No day was read.",
              f"{c['conc']} of the {c['pairs']} comparisons are this between-patient kind.",
              auroc, "AUROC", att)
    _question(ax, 97, 103.0,
              "The within-person C compares a patient with herself", "c",
              ("Patient A", A_ATK_DAY + 1, True, A_RATE),
              ("Patient A", A_QUIET_DAY + 1, False, A_RATE),
              "0.50  =  0.50", False,
              "This is the comparison a patient actually needs, and\nthe forecast is "
              "identical on both days.",
              f"All {c['tied']} within-patient comparisons are ties.",
              0.5, "C", att)

    _brier(ax, bs, bs_ref, skill)
    print("saved", S.save(fig, HERE / "figures" / "fig_a6_pooled_vs_within"))


if __name__ == "__main__":
    main()
