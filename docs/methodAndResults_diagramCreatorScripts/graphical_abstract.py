"""Graphical abstract for the JHP submission.

JHP specification: 920 x 300 px, <=150 KB, JPG/PNG/SVG, file name "Graphical Abstract"
(https://thejournalofheadacheandpain.biomedcentral.com/graphical-abstracts).

One claim, drawn: the same model scored two ways. Pooled AUROC (across patients)
collapses to a near-chance within-person C-statistic. The arrow is the finding;
everything else is support.

Layout discipline: every coordinate is a multiple of the 8 px grid unit G, and the
named Y_/X_ constants below are the only positions used. Earlier revisions placed
elements by eye and read as noise -- shared baselines and even gaps are most of what
makes a figure look deliberate. Keep new elements on the grid.

Two constraints that are correctness, not taste:
  * Orange TEXT uses MIG_T (#C25100, 4.70:1 on white). #D55E00 is 3.87:1 and fails
    the 4.5:1 WCAG floor; it is fine for marks and lines, not for type.
  * The within-person values are pinned ABOVE the chance line (18*G / 14*G). Placed
    by data offset, 0.54 renders below y(0.50) and reads as below-chance, which is
    false.

Numbers trace to article.tex / supplementary_body.tex; see graphical_abstract_review.md
in the paper working tree before changing any value.
"""
import os
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import sys

from PIL import Image

HERE = Path(__file__).resolve().parent

MIG, HEA = "#D55E00", "#0072B2"
MIG_T = "#C25100"
INK, BODY, GREY, FAINT = "#1a1a1a", "#3d3d3d", "#5f5f5f", "#d8d8d8"

W, H = 920, 300
G  = 8                       # grid unit; every coordinate below is a multiple
M  = 6*G                     # 48  page margin, shared by title/body/footer

# --- vertical rhythm, top-down, in whole units -------------------------------
Y_TITLE = H - 4*G            # 268
Y_SUB   = Y_TITLE - 3*G      # 244
Y_HEAD  = Y_SUB   - 4*G      # 212
PY1     = 23*G               # 184  top of scale (0.82)
PY0     = 11*G               # 88   bottom of scale (0.48)
Y_NOTE  = 7*G                # 56
Y_RULE  = 5*G                # 40
Y_FOOT  = 3*G                # 24

# --- horizontal grid ---------------------------------------------------------
X_NAME  = 26*G               # 208  right edge of series name
X_VAL0  = 40*G               # 320  right edge of pooled value
HX0     = 43*G               # 344  pooled marker
HX1     = 75*G               # 600  within marker
X_VAL1  = 80*G               # 640  left edge of within value
X_NEAR  = 93*G               # 744  left edge of "near chance"

A0, A1 = 0.48, 0.82
def y(v): return PY0 + (v-A0)/(A1-A0)*(PY1-PY0)

fig = plt.figure(figsize=(W/100, H/100), dpi=100)
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
ax.add_patch(Rectangle((0,0),W,H,fc="white",ec="none",zorder=-10))

ax.text(M, Y_TITLE, "Pooled AUROC overstates day-to-day usefulness for an individual patient",
        fontsize=14, fontweight="bold", color=INK, va="center", ha="left")
ax.text(M, Y_SUB, "One model, two ways of scoring the same predictions",
        fontsize=10.5, color=GREY, va="center", ha="left")

ax.text(HX0, Y_HEAD, "Pooled across patients", fontsize=11, fontweight="bold",
        color=INK, ha="center", va="center")
ax.text(HX1, Y_HEAD, "Within one patient", fontsize=11, fontweight="bold",
        color=INK, ha="center", va="center")

ax.plot([M+G, 79*G],[y(0.5)]*2, ls=(0,(5,4)), lw=1.0, color="#9a9a9a", zorder=1)
ax.text(M+2*G, y(0.5), "chance  0.50", fontsize=9.5, color=GREY, va="center", ha="left",
        bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none"))

DATA = [("migraine", MIG, MIG_T, 0.791,(0.544,0.890), 0.558,(0.511,0.604), "0.23",  1),
        ("headache", HEA, HEA,   0.653,(0.555,0.740), 0.542,(0.509,0.575), "0.11", -1)]
for name,c,ct,p,pci,w,wci,d,side in DATA:
    xo = side*G                                   # one grid unit, not an eyeball
    for xx,ci in ((HX0+xo,pci),(HX1+xo,wci)):
        ax.plot([xx,xx],[y(ci[0]),y(ci[1])], lw=1.2, color=c, alpha=.33, zorder=2)
        for e in ci: ax.plot([xx-4,xx+4],[y(e)]*2, lw=1.2, color=c, alpha=.33, zorder=2)
    ax.add_patch(FancyArrowPatch((HX0+xo,y(p)),(HX1+xo,y(w)), arrowstyle="-|>",
                 mutation_scale=15, lw=3.0, color=c, shrinkA=9, shrinkB=9, zorder=3))
    for xx,v in ((HX0+xo,p),(HX1+xo,w)):
        ax.plot([xx],[y(v)],"o",ms=10,color=c,zorder=4)
    ax.text(X_VAL0, y(p), f"{p:.2f}", fontsize=21, fontweight="bold", color=ct,
            ha="right", va="center")
    ax.text(X_NAME, y(p), name, fontsize=10.5, fontweight="bold", color=ct,
            ha="right", va="center")
    ax.text(X_VAL1, 18*G if side > 0 else 14*G, f"{w:.2f}", fontsize=21,
            fontweight="bold", color=ct, ha="left", va="center")
    ax.text((HX0+HX1)/2, (y(p)+y(w))/2 + side*2*G, f"−{d}", fontsize=13, fontweight="bold",
            color=ct, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=FAINT, lw=0.9))

ax.text(X_NEAR, 16*G, "near chance", fontsize=10, color=GREY, va="center", ha="left")

# two aligned facts, not one prose sentence
ax.text(M,    Y_NOTE, "Persists on onset-only days and in unseen patients",
        fontsize=9.5, color=BODY, va="center", ha="left")
ax.text(60*G, Y_NOTE, "Alarms on 98–100% of days — breaches ICHD-3 ceilings",
        fontsize=9.5, color=BODY, va="center", ha="left")

ax.plot([M, W-M],[Y_RULE]*2, lw=0.9, color=FAINT)
ax.text(M, Y_FOOT, "62 patients · 4,516 diary days · Park 2016 Korean smartphone-diary cohort",
        fontsize=9.5, color=GREY, va="center", ha="left")

out_dir = HERE / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_png = out_dir / "graphical_abstract.png"
fig.savefig(out_png, dpi=100, bbox_inches=None, facecolor="white")
plt.close(fig)

with Image.open(out_png) as im:
    iw, ih = im.size
    im.convert("RGB").save(out_png, optimize=True)
size_kb = os.path.getsize(out_png) / 1024.0
print(f"saved {out_png.name}  {iw}x{ih} px, {size_kb:.1f} KB")
if (iw, ih) != (W, H) or size_kb > 150:
    print("WARNING: violates JHP spec (920x300 px, <=150 KB)", file=sys.stderr)
    sys.exit(1)
