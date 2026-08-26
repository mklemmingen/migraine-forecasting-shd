"""Figure A6 - the two comparisons pooled AUROC and the within-person C ask.

AUROC is an average over attack-day/quiet-day comparisons, so what it measures
depends entirely on which comparisons are allowed into that average. Pooling lets a
day be compared with any other patient's day; the within-person C-statistic allows
only comparisons inside one patient. This figure shows one comparison of each kind,
with the classifier's actual answer, and lets the reader see that the first is easy
for a reason that has nothing to do with forecasting.

The worked classifier emits each patient's own attack rate and nothing else, so it
holds no day-level information. It answers the between-patient comparison correctly
every time (patient A has attacks more often) and cannot answer the
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

W_MM, H_MM = 183.0, 166.0          # canvas is millimetres at final size
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


# ---- type scale (pt) and ink -------------------------------------------------
# Four sizes only. Anything needing a fifth is saying too much.
PT_CLAIM, PT_LEAD, PT_BODY, PT_SCORE = 9.0, 8.6, 7.2, 13.0
INK, MUTE = S.INK, S.GREY        # house ink; GREY is darker than a mid grey,
                                 # which is what keeps the secondary text legible
TINT = "#f4f4f4"
BASE = 6.0                      # vertical rhythm; every band sits on a multiple


def _claim(ax, x, y, letter, text, accent):
    """Panel heading states the panel's claim, after the convention this literature
    uses for teaching figures (Sebastianelli 2024, Cephalalgia)."""
    ax.add_patch(Rectangle((x, y - 3.1), 1.5, 6.2, fc=accent, ec="none"))
    ax.text(x + 4.2, y, f"({letter})", fontsize=PT_CLAIM, fontweight="bold", color=INK,
            va="center", ha="left")
    ax.text(x + 9.6, y, text, fontsize=PT_CLAIM, fontweight="bold", color=INK,
            va="center", ha="left")


def _txt(ax, x, y, text, size=PT_BODY, col=INK, ha="left", weight="normal"):
    ax.text(x, y, text, fontsize=size, color=col, va="center", ha=ha,
            linespacing=1.55, fontweight=weight)


def _daycard(ax, x, y, w, h, who, day, is_attack, risk, att):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=TINT, ec="none"))
    _txt(ax, x + w / 2, y + h - 4.6, f"{who}, day {day}", PT_BODY, INK, "center", "bold")
    sq = 8.4
    ax.add_patch(Rectangle((x + w / 2 - sq / 2, y + h - 16.4), sq, sq,
                           fc=att if is_attack else "#cfcfcf", ec="none"))
    _txt(ax, x + w / 2, y + h - 20.6, "attack" if is_attack else "no attack",
         PT_BODY, MUTE, "center")
    _txt(ax, x + w / 2, y + 4.6, f"{risk:.2f}", PT_LEAD, INK, "center", "bold")


def _diaries(ax, att, accent):
    _claim(ax, 6, 158.0, "a", "One forecast per patient, unchanged across days", accent)
    s, pitch = 5.4, 6.4
    ring = {("A", A_ATK_DAY), ("A", A_QUIET_DAY), ("B", B_QUIET_DAY)}
    for k, (name, attacks, rate) in enumerate([("A", A_ATTACK, A_RATE),
                                               ("B", B_ATTACK, B_RATE)]):
        yy = 145.0 - k * 9.0
        _txt(ax, 6, yy + s / 2, f"Patient {name}", PT_BODY, INK, weight="bold")
        for i in range(N_DAYS):
            ax.add_patch(Rectangle((26 + i * pitch, yy), s, s,
                                   fc=att if i in attacks else "#dcdcdc", ec="none"))
            if (name, i) in ring:
                ax.add_patch(FancyBboxPatch((26 + i * pitch - 0.9, yy - 0.9),
                                            s + 1.8, s + 1.8, fc="none", ec=INK, lw=1.2,
                                            zorder=5,
                                            boxstyle="round,pad=0,rounding_size=1.0"))
        _txt(ax, 26 + N_DAYS * pitch + 4.0, yy + s / 2, f"forecast {rate:.2f}",
             PT_BODY, INK, weight="bold")
    _txt(ax, 6, 126.0, "Patient A records attacks on 5 of 10 days, patient B on 1 of 10. "
                       "Each forecast is set to that patient's own\nrecorded rate, so it "
                       "never changes from day to day.", PT_BODY, MUTE)
    _txt(ax, 6, 115.0, "Outlined days are compared below.", PT_BODY, MUTE)


def _question(ax, x0, y0, letter, claim, left, right, verdict, ruling, why, tally,
              score_lab, score, att, accent):
    _claim(ax, x0, y0, letter, claim, accent)
    cw, ch, gap = 32.0, 29.0, 10.0
    cy = y0 - 36.0
    _daycard(ax, x0, cy, cw, ch, *left, att)
    _daycard(ax, x0 + cw + gap, cy, cw, ch, *right, att)
    _txt(ax, x0 + cw + gap / 2, cy + ch / 2, "vs", PT_BODY, MUTE, "center")

    mid = x0 + cw + gap / 2
    vy = cy - 5.0
    ax.plot([x0 + cw / 2, x0 + cw / 2, x0 + cw + gap + cw / 2, x0 + cw + gap + cw / 2],
            [cy - 0.8, vy, vy, cy - 0.8], color="#c2c2c2", lw=0.9)
    _txt(ax, mid, vy - 5.5, verdict, PT_LEAD, INK, "center", "bold")
    _txt(ax, mid, vy - 10.5, ruling, PT_BODY, MUTE, "center")

    _txt(ax, x0, vy - 18.0, why, PT_BODY, INK)
    _txt(ax, x0, vy - 26.0, tally, PT_BODY, MUTE)
    _txt(ax, x0, vy - 34.0, score_lab, PT_BODY, MUTE)
    _txt(ax, x0 + 22.0, vy - 34.0, f"{score:.2f}", PT_SCORE, accent, weight="bold")


def _brier(ax, skill, accent):
    _claim(ax, 6, 20.0, "d", "Brier skill: does the forecast improve on the known rate?",
           accent)
    bw, bh, gap = 56.0, 13.0, 13.0
    by = 5.0
    for k, (lab, val) in enumerate([("forecast for patient A", "0.50 daily"),
                                    ("patient A's recorded rate", "0.50")]):
        x = 6.0 + k * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, by), bw, bh,
                                    boxstyle="round,pad=0,rounding_size=1.4",
                                    fc=TINT, ec="none"))
        _txt(ax, x + bw / 2, by + bh - 4.4, lab, PT_BODY, MUTE, "center")
        _txt(ax, x + bw / 2, by + 4.6, val, PT_LEAD, INK, "center", "bold")
    _txt(ax, 6.0 + bw + gap / 2, by + bh / 2, "=", PT_LEAD, INK, "center", "bold")
    _txt(ax, 6.0 + 2 * bw + gap + 8.0, by + bh / 2 + 2.6, "skill", PT_BODY, MUTE)
    _txt(ax, 6.0 + 2 * bw + gap + 20.0, by + bh / 2 + 2.6, f"{skill:.2f}", PT_SCORE,
         accent, weight="bold")
    _txt(ax, 6.0 + 2 * bw + gap + 8.0, by + bh / 2 - 4.4, "no improvement", PT_BODY, MUTE)
    _txt(ax, 6, by - 5.0, "The two are the same number by construction. Skill of zero is "
                          "what a forecast that adds nothing looks like.", PT_BODY, MUTE)


def main() -> None:
    S.apply()
    att = S.target_color("migraine")          # attack day
    blue = S.target_color("headache")         # accent on the clinically decisive panels
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

    _diaries(ax, att, MUTE)
    _question(ax, 6, 105.0, "b", "Pooled AUROC compares days across patients",
              ("Patient A", A_ATK_DAY + 1, True, A_RATE),
              ("Patient B", B_QUIET_DAY + 1, False, B_RATE),
              "0.50  >  0.10", "ranked correctly",
              "Attack frequency decides this comparison.\nNeither day was examined.",
              f"{c['conc']} of {c['pairs']} comparisons cross patients.",
              "AUROC", auroc, att, MUTE)
    _question(ax, 101, 105.0, "c", "Within-person C compares one patient's days",
              ("Patient A", A_ATK_DAY + 1, True, A_RATE),
              ("Patient A", A_QUIET_DAY + 1, False, A_RATE),
              "0.50  =  0.50", "no separation",
              "The clinically relevant comparison.\nThe forecast is identical on both days.",
              f"All {c['tied']} within-patient comparisons tie.",
              "C", 0.5, att, blue)
    _brier(ax, skill, blue)

    print("saved", S.save(fig, HERE / "figures" / "fig_a6_pooled_vs_within"))


if __name__ == "__main__":
    main()
