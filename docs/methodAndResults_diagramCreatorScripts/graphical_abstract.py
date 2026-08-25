"""Graphical abstract for the JHP submission.

JHP specification: 920 x 300 px, <=150 KB, JPG/PNG/SVG, file name "Graphical Abstract"
(https://thejournalofheadacheandpain.biomedcentral.com/graphical-abstracts).

Design follows "Ten simple rules for designing graphical abstracts"
(PLoS Comput Biol; PMC10833524): one load-bearing claim in the title, a single
hero comparison read left-to-right, arrows as the connector that carries the
message, and no axis furniture that the reader does not need.

Layout:
  Hero (left, ~560 px): paired slopegraph, pooled AUROC -> within-person C, one
    slope per outcome. The arrow IS the finding. Endpoint CIs are offset capped
    whiskers so the two series never merge into a false axis spine. Series names
    are set in their own colour and act as the legend.
  Evidence rail (right, ~330 px): three supporting blocks in reading order --
    the gap is not a split artefact, what drives the model, why it is not
    deployable. Deliberately de-saturated: at thumbnail size the rail must not
    out-shout the hero.

Numbers are traceable to article.tex / supplementary_body.tex; see the
verification notes in the paper working files before editing any value.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent

# Okabe-Ito vermillion / blue. MIG_T is the darkened orange used for TEXT only:
# #D55E00 is 3.87:1 on white and fails the 4.5:1 WCAG floor; #C25100 is 4.70:1.
MIG, HEA = "#D55E00", "#0072B2"
MIG_T = "#C25100"
INK, SOFT, FAINT, GREY = "#222222", "#444444", "#cccccc", "#555555"
MUTE_M, MUTE_H = "#e8c4ad", "#b9d4e8"

W, H = 920, 300

# Pooled AUROC and within-person C-statistic, both with patient-CLUSTER CIs
# (not patient-day). article.tex Table 2.
MIGRAINE = dict(pooled=0.791, pooled_ci=(0.544, 0.890),
                within=0.558, within_ci=(0.511, 0.604), delta="0.23")
HEADACHE = dict(pooled=0.653, pooled_ci=(0.555, 0.740),
                within=0.542, within_ci=(0.509, 0.575), delta="0.11")


def main() -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=-10))

    ax.text(24, 286,
            "Pooled AUROC overstates day-to-day usefulness for an individual patient",
            fontsize=12.5, fontweight="bold", color=INK, va="center", ha="left")

    HX0, HX1 = 178, 424
    PY0, PY1, A0, A1 = 96, 224, 0.46, 0.84

    def y(v):
        return PY0 + (v - A0) / (A1 - A0) * (PY1 - PY0)

    # Chance is a LINE, not a shaded band: a band placed 0.558 outside and 0.542
    # inside, inventing a categorical split between two near-identical results.
    ax.plot([96, 520], [y(0.5)] * 2, ls=(0, (4, 3)), lw=1.0, color=GREY, zorder=1)
    ax.text(96, y(0.5) - 12, "chance  0.50", fontsize=8.5, color=GREY,
            va="center", ha="left")

    # Headers name the two scoring METHODS. Without this the slope reads as a
    # change over time rather than two ways of scoring one model.
    ax.text(HX0, 261, "Pooled\nacross patients", fontsize=10, fontweight="bold",
            color=INK, ha="center", va="center", linespacing=1.15)
    ax.text(HX1, 261, "Within one patient\n(within-person C)", fontsize=10,
            fontweight="bold", color=INK, ha="center", va="center", linespacing=1.15)
    ax.text((HX0 + HX1) / 2, 66, "same model, two ways of scoring", fontsize=8.5,
            style="italic", color=GREY, ha="center", va="center")

    series = [("migraine", MIG, MIG_T, MIGRAINE, 1),
              ("headache", HEA, HEA, HEADACHE, -1)]
    for name, c, ct, d, side in series:
        xo = 9 * side  # per-series offset so whiskers and markers never merge
        for xx, val, ci in ((HX0 - 34 + xo, d["pooled"], d["pooled_ci"]),
                            (HX1 + 34 + xo, d["within"], d["within_ci"])):
            ax.plot([xx, xx], [y(ci[0]), y(ci[1])], lw=1.1, color=c, alpha=.45, zorder=2)
            for e in ci:
                ax.plot([xx - 3.5, xx + 3.5], [y(e)] * 2, lw=1.1, color=c, alpha=.45, zorder=2)
        ax.add_patch(FancyArrowPatch((HX0 - 34 + xo, y(d["pooled"])),
                                     (HX1 + 34 + xo, y(d["within"])),
                                     arrowstyle="-|>", mutation_scale=12, lw=2.6,
                                     color=c, shrinkA=6, shrinkB=6, zorder=3))
        ax.plot([HX0 - 34 + xo], [y(d["pooled"])], "o", ms=9, color=c, zorder=4)
        ax.plot([HX1 + 34 + xo], [y(d["within"])], "o", ms=9, color=c, zorder=4)
        ax.text(HX0 - 52 + xo, y(d["pooled"]), f"{d['pooled']:.2f}", fontsize=16,
                fontweight="bold", color=ct, ha="right", va="center")
        ax.text(HX1 + 52 + xo, y(d["within"]) + 13 * side, f"{d['within']:.2f}",
                fontsize=16, fontweight="bold", color=ct, ha="left", va="center")
        ax.text(HX0 - 52 + xo, y(d["pooled"]) + 16 * side, name, fontsize=9,
                fontweight="bold", color=ct, ha="right", va="center")
        mx, my = (HX0 + HX1) / 2, (y(d["pooled"]) + y(d["within"])) / 2
        ax.text(mx, my + 14 * side, f"−{d['delta']}", fontsize=11.5, fontweight="bold",
                color=ct, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=FAINT, lw=0.7))

    # One bracket groups BOTH endpoints into a single conclusion.
    gx = HX1 + 104
    ax.plot([gx, gx + 6, gx + 6, gx],
            [y(0.508), y(0.508), y(0.60), y(0.60)], lw=0.9, color=GREY)
    ax.text(gx + 10, y(0.554), "near\nchance", fontsize=8, color=GREY,
            va="center", ha="left", linespacing=1.1)

    RX = 588
    ax.plot([RX - 22, RX - 22], [50, 250], lw=0.8, color=FAINT)
    ax.text(RX, 244, "Not a split artefact", fontsize=9.5, fontweight="bold",
            color=INK, va="top")
    for i, t in enumerate(["Onset-only days: 0.79 → 0.76",
                           "New patients, same cell: pooled",
                           "0.62 → 0.31 (below chance);",
                           "within-person holds 0.56 → 0.51"]):
        ax.text(RX, 228 - i * 13, t, fontsize=8.5, color=SOFT, va="top")
    ax.plot([RX, 898], [168] * 2, lw=0.7, color=FAINT)

    ax.text(RX, 158, "Driven by history, not triggers", fontsize=9.5,
            fontweight="bold", color=INK, va="top")
    # Bars are de-saturated on purpose: saturated 90%/77% bars were the only
    # colour on the right at thumbnail size, so a skimmer read "0.79 and 90% =
    # good" and inverted the message.
    for i, (frac, cm, lab) in enumerate([(0.90, MUTE_M, "90%  migraine"),
                                         (0.77, MUTE_H, "77%  headache")]):
        yy = 136 - i * 16
        ax.add_patch(Rectangle((RX, yy), 190, 9, fc="#f0f0f0", ec="none"))
        ax.add_patch(Rectangle((RX, yy), 190 * frac, 9, fc=cm, ec="none"))
        ax.text(RX + 196, yy + 4.5, lab, fontsize=8.5, color=SOFT, va="center")
    ax.text(RX, 112, "of SHAP attribution from 26 engineered", fontsize=8.5,
            color=SOFT, va="top")
    ax.text(RX, 99, "history features", fontsize=8.5, color=SOFT, va="top")

    ax.text(RX, 84, "Not deployable", fontsize=9.5, fontweight="bold",
            color=MIG_T, va="top")
    ax.text(RX, 70, "Alarms on 98–100% of days; all 20", fontsize=8.5, color=SOFT, va="top")
    ax.text(RX, 57, "patients breach ICHD-3 ceilings", fontsize=8.5, color=SOFT, va="top")

    ax.plot([24, 898], [42] * 2, lw=0.8, color=FAINT)
    ax.text(24, 26,
            "62 patients · 4,516 diary days · migraine on 7.2% of days · "
            "Park 2016 Korean smartphone-diary cohort",
            fontsize=8.5, color=GREY, va="center", ha="left")

    out_dir = HERE / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "graphical_abstract.png"
    fig.savefig(out_png, dpi=100, bbox_inches=None, facecolor="white")
    plt.close(fig)

    with Image.open(out_png) as im:
        w, h = im.size
        im.convert("RGB").save(out_png, optimize=True)
    size_kb = os.path.getsize(out_png) / 1024.0
    print(f"saved {out_png.name}  {w}x{h} px, {size_kb:.1f} KB")
    if (w, h) != (W, H) or size_kb > 150:
        print("WARNING: violates JHP spec (920x300 px, <=150 KB)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
