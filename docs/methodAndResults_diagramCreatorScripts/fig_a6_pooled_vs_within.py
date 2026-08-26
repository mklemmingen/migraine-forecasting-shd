"""Figure A6 - what pooled AUROC, the within-person C-statistic, and Brier skill count.

AUROC is a rank statistic over attack-day/quiet-day pairs, so this figure draws the
pairs. With 6 attack days and 14 quiet days there are exactly 84 of them: few enough
that every pair gets its own mark and nothing has to be asserted in prose.

The worked cohort is a classifier that emits each patient's own attack rate and
nothing else, so it carries no day-level information at all. Ordering the matrix by
patient then makes the mechanism arrange itself: the same-patient pairs land in the
two block-diagonal cells, and because the forecast is constant inside a patient every
one of those 34 pairs is tied. The within-person restriction is therefore not a region
someone chose to outline, it is where the arithmetic already put the ties.

The same classifier is the per-patient climatology this paper uses as the Brier
reference, which fixes its Brier skill at exactly zero. One classifier reads 0.74
pooled, 0.50 within patient, 0.00 against the patient's own rate.

Every count and all three statistics are derived from the toy arrays at draw time and
checked against sklearn, so the figure cannot disagree with itself.

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
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent

# Canvas is in millimetres at final size: aspect is equal and the figure is sized
# to match, so one data unit is one mm on the printed page.
W_MM, H_MM = 183.0, 146.0
QUIET = "#e3e3e3"
RULE = "#9a9a9a"

N_DAYS = 10
A_ATTACK, B_ATTACK = [0, 2, 5, 6, 9], [4]
A_RATE, B_RATE = len(A_ATTACK) / N_DAYS, len(B_ATTACK) / N_DAYS


def _toy():
    y = np.array([1 if i in A_ATTACK else 0 for i in range(N_DAYS)]
                 + [1 if i in B_ATTACK else 0 for i in range(N_DAYS)])
    p = np.array([A_RATE] * N_DAYS + [B_RATE] * N_DAYS)
    return y, p


def _pairs():
    """Every attack/quiet pair, ordered by patient so the blocks fall out on their own."""
    atk = [("A", A_RATE)] * len(A_ATTACK) + [("B", B_RATE)] * len(B_ATTACK)
    qui = [("A", A_RATE)] * (N_DAYS - len(A_ATTACK)) + [("B", B_RATE)] * (N_DAYS - len(B_ATTACK))
    grid = []
    for pa, sa in atk:
        row = []
        for pq, sq in qui:
            row.append(("tied" if sa == sq else ("conc" if sa > sq else "disc"), pa == pq))
        grid.append(row)
    return atk, qui, grid


def _cell(ax, x, y, s, kind, att, hea):
    """One pair. State is carried by FILL PATTERN, not hue, so the matrix survives
    greyscale and colour-blind reading; the hues are reinforcement only."""
    if kind == "conc":
        ax.add_patch(Rectangle((x, y), s, s, fc=att, ec="none"))
    elif kind == "disc":
        ax.add_patch(Rectangle((x, y), s, s, fc=hea, ec="none"))
        ax.plot([x + s * .22, x + s * .78], [y + s * .22, y + s * .78], color="white", lw=0.7)
        ax.plot([x + s * .22, x + s * .78], [y + s * .78, y + s * .22], color="white", lw=0.7)
    else:
        ax.add_patch(Rectangle((x, y), s, s, fc="white", ec=RULE, lw=0.55))
        ax.plot([x + s * .26, x + s * .74], [y + s * .40, y + s * .40], color=RULE, lw=0.55)
        ax.plot([x + s * .26, x + s * .74], [y + s * .60, y + s * .60], color=RULE, lw=0.55)


def _head(ax, x, y, letter, text, size=8.4):
    ax.text(x, y, f"({letter})", fontsize=size, fontweight="bold", color=S.INK,
            va="center", ha="left")
    ax.text(x + 5.4, y, text, fontsize=size, fontweight="bold", color=S.INK,
            va="center", ha="left")


def _note(ax, x, y, text, size=7.0, col="#3d3d3d", ha="left", box=None):
    bb = dict(fc=box, ec="none", pad=1.4) if box else None
    ax.text(x, y, text, fontsize=size, color=col, va="center", ha=ha, linespacing=1.5,
            bbox=bb, zorder=8)


def _cohort(ax, att):
    """The classifier: one number per patient, repeated across all ten of their days."""
    _head(ax, 4, 141.0, "a", "A classifier that emits each patient's own attack rate")
    s, pitch = 5.0, 5.9
    for k, (name, attacks, rate) in enumerate(
            [("A", A_ATTACK, A_RATE), ("B", B_ATTACK, B_RATE)]):
        yy = 132.0 - k * 7.4
        ax.text(4, yy + s / 2, name, fontsize=7.4, fontweight="bold", color=S.INK,
                va="center", ha="left")
        for i in range(N_DAYS):
            ax.add_patch(Rectangle((10 + i * pitch, yy), s, s,
                                   fc=att if i in attacks else QUIET, ec="none"))
        x_end = 10 + N_DAYS * pitch
        ax.plot([x_end + 1.2, x_end + 2.4, x_end + 2.4, x_end + 1.2],
                [yy, yy, yy + s, yy + s], color=RULE, lw=0.7)
        ax.text(x_end + 4.0, yy + s / 2, f"forecast {rate:.2f}\non every day",
                fontsize=6.6, color="#3d3d3d", va="center", ha="left", linespacing=1.35)
    _note(ax, 4, 116.0, "The bracket is the point: one value covers"
                       "\nall ten days, so the forecast says nothing"
                       "\nabout which day an attack falls on.", col="#5f5f5f")


def _matrix(ax, grid, att, hea):
    """All 84 pairs. Rows are attack days, columns quiet days, both ordered by patient,
    so the two same-patient blocks land on the diagonal without being placed there."""
    X0, Y0, s, g = 96.0, 96.0, 5.1, 0.9      # Y0 is the BOTTOM of the matrix
    pitch = s + g
    nr, nc = len(grid), len(grid[0])
    _head(ax, 96, 141.0, "b", "Every attack-day / quiet-day pair, one square each")

    for r, row in enumerate(grid):
        for c, (kind, _same) in enumerate(row):
            _cell(ax, X0 + c * pitch, Y0 + (nr - 1 - r) * pitch, s, kind, att, hea)

    nA_atk, nA_qui = len(A_ATTACK), N_DAYS - len(A_ATTACK)
    # column / row spans naming which patient each band belongs to
    ax.plot([X0, X0 + nA_qui * pitch - g], [Y0 + nr * pitch + 0.8] * 2, color=RULE, lw=0.7)
    ax.plot([X0 + nA_qui * pitch, X0 + nc * pitch - g], [Y0 + nr * pitch + 0.8] * 2,
            color=RULE, lw=0.7)
    _note(ax, X0 + nA_qui * pitch / 2, Y0 + nr * pitch + 3.0, "A's 5 quiet days",
          size=6.4, col="#5f5f5f", ha="center")
    _note(ax, X0 + (nA_qui + (nc - nA_qui) / 2) * pitch, Y0 + nr * pitch + 3.0,
          "B's 9 quiet days", size=6.4, col="#5f5f5f", ha="center")
    ax.plot([X0 - 1.4] * 2, [Y0 + (nr - nA_atk) * pitch, Y0 + nr * pitch - g],
            color=RULE, lw=0.7)
    ax.plot([X0 - 1.4] * 2, [Y0, Y0 + (nr - nA_atk) * pitch - g], color=RULE, lw=0.7)
    _note(ax, X0 - 2.6, Y0 + (nr - nA_atk / 2) * pitch - g / 2, "A's 5\nattack days",
          size=6.4, col="#5f5f5f", ha="right")
    _note(ax, X0 - 2.6, Y0 + pitch / 2, "B's 1", size=6.4, col="#5f5f5f", ha="right")

    # the within-person restriction: exactly the two same-patient blocks
    for (r0, nr_b, c0, nc_b) in [(0, nA_atk, 0, nA_qui),
                                 (nA_atk, nr - nA_atk, nA_qui, nc - nA_qui)]:
        x = X0 + c0 * pitch - g / 2
        yb = Y0 + (nr - r0 - nr_b) * pitch - g / 2
        ax.add_patch(FancyBboxPatch((x, yb), nc_b * pitch, nr_b * pitch,
                                    boxstyle="round,pad=0.35,rounding_size=1.0",
                                    fc="none", ec=S.INK, lw=1.3, zorder=6))
    # block counts, placed inside their own blocks
    _note(ax, X0 + nA_qui * pitch / 2, Y0 + (nr - nA_atk / 2) * pitch - g / 2,
          "25 tied", size=7.0, col=S.INK, ha="center", box="white")
    _note(ax, X0 + (nA_qui + (nc - nA_qui) / 2) * pitch, Y0 + (nr - nA_atk / 2) * pitch - g / 2,
          "45 concordant", size=7.4, col=S.INK, ha="center", box="white")
    _note(ax, X0 + nA_qui * pitch / 2, Y0 - 3.4, "5 discordant", size=6.6,
          col=S.INK, ha="center")
    _note(ax, X0 + (nA_qui + (nc - nA_qui) / 2) * pitch, Y0 - 3.4, "9 tied",
          size=6.6, col=S.INK, ha="center")
    return X0, Y0, s, pitch, nr, nc


def _rails(ax, counts, auroc, att, hea):
    """The same 84 squares restacked into one ordered row. The score is a POSITION
    along that row, so 'ties count one half' is where the pointer lands, not a formula."""
    conc, disc, tied, tot = counts
    _head(ax, 4, 78.0, "c", "The same squares, restacked: each score is a position")
    s, g = 1.68, 0.34
    pitch = s + g
    for lab, segs, n_tot, score, won, y in [
            ("pooled", [("conc", conc), ("tied", tied), ("disc", disc)], tot, auroc, conc, 66.0),
            ("within person", [("tied", tied)], tied, 0.5, 0, 51.0)]:
        x = 11.0
        for kind, n in segs:
            for _ in range(n):
                _cell(ax, x, y, s, kind, att, hea)
                x += pitch
        _note(ax, 9.0, y + s / 2, lab, size=7.2, col=S.INK, ha="right")
        _note(ax, 11.0 + n_tot * pitch + 1.6, y + s / 2, f"{n_tot} pairs", size=6.6,
              col="#5f5f5f")
        pos = 11.0 + (won + 0.5 * tied) * pitch
        ax.add_patch(Polygon([[pos, y - 0.7], [pos - 1.4, y - 3.4], [pos + 1.4, y - 3.4]],
                             closed=True, fc=S.INK, ec="none"))
        _note(ax, pos, y - 5.6, f"{score:.2f}", size=8.8, col=S.INK, ha="center")
    _note(ax, 4, 39.5, "The pointer sits at the concordant squares plus half the tied ones. "
                      "Within a patient there are\nno concordant squares at all, so it lands "
                      "mid-tie: chance.", size=6.8, col="#5f5f5f")


def _legend(ax, att, hea):
    x, y = 96.0, 81.5
    for kind, lab in [("conc", "concordant"), ("disc", "discordant"), ("tied", "tied")]:
        _cell(ax, x, y, 3.4, kind, att, hea)
        _note(ax, x + 4.6, y + 1.7, lab, size=6.6, col="#5f5f5f")
        x += 27.0


def _brier(ax, y_true, p, ref, bs, bs_ref, skill, hea):
    """Brier is not a pair statistic, so it gets its own device: each day's squared
    error drawn as an actual square of that area. The forecast row and the own-rate row
    are drawn as two aligned rows; they are square-for-square identical, which is the
    whole finding, and identity is easier to see as congruence than as coincidence."""
    _head(ax, 4, 31.0, "d", "Brier skill: squared error against the patient's own rate")
    x0, step, k = 34.0, 7.4, 6.0
    for row, (vals, filled, lab) in enumerate(
            [(p, True, "forecast"), (ref, False, "patient's own rate")]):
        ymid = 24.0 - row * 8.4
        _note(ax, x0 - 3.0, ymid, lab, size=6.6, col="#5f5f5f", ha="right")
        for i, (yt, vi) in enumerate(zip(y_true, vals)):
            side = abs(vi - yt) * k
            cx = x0 + i * step
            if filled:
                ax.add_patch(Rectangle((cx - side / 2, ymid - side / 2), side, side,
                                       fc=hea, ec="none", alpha=0.75))
            else:
                ax.add_patch(Rectangle((cx - side / 2, ymid - side / 2), side, side,
                                       fc="none", ec=S.INK, lw=0.7))
    _note(ax, 4, 10.0, "One square per diary day, its area the squared error. The two rows "
                        "are identical square for\nsquare, because the forecast IS that rate.",
          size=6.8, col="#5f5f5f")
    ax.text(4, 3.4, rf"$\mathrm{{skill}}=1-\frac{{{bs:.2f}}}{{{bs_ref:.2f}}}={skill:.2f}$",
            fontsize=9.4, color=S.INK, va="center", ha="left")
    _note(ax, 40, 3.4, "the forecast adds nothing to that rate", size=7.0)


def main() -> None:
    S.apply()
    plt.rcParams["mathtext.default"] = "regular"
    att, hea = S.target_color("migraine"), S.target_color("headache")
    y, p = _toy()
    ref = np.array([A_RATE] * N_DAYS + [B_RATE] * N_DAYS)
    auroc = float(roc_auc_score(y, p))
    bs, bs_ref = float(np.mean((p - y) ** 2)), float(np.mean((ref - y) ** 2))
    skill = 1.0 - bs / bs_ref
    _atk, _qui, grid = _pairs()
    flat = [k for row in grid for k, _ in row]
    conc, disc, tied = flat.count("conc"), flat.count("disc"), flat.count("tied")
    tot = len(flat)
    same = sum(1 for row in grid for _k, sm in row if sm)

    assert conc + disc + tied == tot == 84, "pair counts must partition 84"
    assert same == tied, "the same-patient block must be exactly the tied pairs"
    assert abs(auroc - (conc + 0.5 * tied) / tot) < 1e-12, "matrix must equal sklearn AUROC"
    assert abs(skill) < 1e-12, "climatology classifier must have exactly zero Brier skill"

    fig, ax = plt.subplots(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM)
    ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    _cohort(ax, att)
    _matrix(ax, grid, att, hea)
    _legend(ax, att, hea)
    _rails(ax, (conc, disc, tied, tot), auroc, att, hea)
    _brier(ax, y, p, ref, bs, bs_ref, skill, hea)

    print("saved", S.save(fig, HERE / "figures" / "fig_a6_pooled_vs_within"))


if __name__ == "__main__":
    main()
