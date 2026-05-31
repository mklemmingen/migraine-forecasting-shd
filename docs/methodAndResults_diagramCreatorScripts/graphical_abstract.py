"""Graphical abstract for the JHP submission.

JHP specification verified against the BMC house guidelines page (May 2026):
  - 920 x 300 px, <=150 KB
  - JPG / PNG / SVG accepted
  - Submitted as a supplementary file alongside the manuscript

Content: a single horizontal bar chart showing the four anchor numbers that
define the paper's title-claim load-bearing finding (pooled AUROC vs
within-person C, for migraine and headache targets), with a chance reference
line at 0.5, the cohort context in the title, and a one-line takeaway naming
the load-bearing finding at the bottom. The four bars carry the exact same
values cited in body §3.2 (pooled AUROC) and §3.6 (within-person C) so the
graphical abstract is reproducible from the body prose and the headline-cell
CSVs without an intermediate transcription step.

Usage: python graphical_abstract.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiment"))
import _style as S  # noqa: E402


def main() -> None:
    S.apply()
    # _style.apply() sets savefig.bbox='tight' which trims margins and breaks the
    # exact-pixel-dimensions requirement of the JHP graphical-abstract spec; we
    # override locally so the saved PNG is exactly figsize x dpi = 920x300 px.
    plt.rcParams["savefig.bbox"] = "standard"

    # 920 x 300 px target at 100 dpi. We render slightly larger than spec then
    # PIL-resize to the exact JHP-mandated dimensions, since matplotlib's
    # subpixel rounding at figsize ~ 9.20 in is non-monotonic and unreliable
    # for hitting an exact-pixel-count target across machines.
    fig, ax = plt.subplots(figsize=(9.21, 3.00), dpi=100)

    # Four bars, top to bottom: migraine pooled / migraine within-person /
    # headache pooled / headache within-person. Values match body §3.2 + §3.6.
    rows = [
        ("migraine pooled AUROC",         0.793, "migraine", 1.00),
        ("migraine within-person C",      0.558, "migraine", 0.40),
        ("headache pooled AUROC",         0.652, "headache", 1.00),
        ("headache within-person C",      0.542, "headache", 0.40),
    ]
    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows])
    colors = [S.target_color(r[2]) for r in rows]
    alphas = [r[3] for r in rows]

    y_pos = np.arange(len(rows))
    for y, v, c, a in zip(y_pos, values, colors, alphas):
        ax.barh(y, v, color=c, alpha=a, edgecolor=S.INK, linewidth=0.6)
        ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=9,
                color=S.INK, fontweight="bold")

    # Chance reference at 0.5 (the load-bearing comparison anchor); label
    # offset right of the dashed line so the "chance" word does not visually
    # merge with the dashed strokes at thumbnail size.
    ax.axvline(x=0.5, color=S.INK, linestyle="--", linewidth=0.9, alpha=0.55)
    ax.text(0.53, -0.55, "chance 0.5", ha="left", va="bottom",
            fontsize=9, color=S.INK, alpha=0.85, fontweight="bold")

    # Y axis: discrimination metric labels, top-down
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()  # top row = migraine pooled (the highest-value bar)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("discrimination (AUROC pooled / Hanley-McNeil within-person C)",
                  fontsize=8.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Title (the load-bearing claim + cohort context)
    ax.set_title(
        "Within-person versus pooled discrimination on next-day migraine / headache forecasting\n"
        "Park 2016 Korean SHD cohort  -  n = 62 patients, 4,516 patient-days, diary-only",
        fontsize=10, loc="left", pad=8,
    )

    # Bottom takeaway (the load-bearing field implication)
    fig.text(
        0.5, 0.015,
        "Pooled AUROC reflects between-patient base-rate separation, "
        "not within-patient day-to-day ranking.",
        ha="center", fontsize=8.5, style="italic", color=S.INK,
    )

    fig.subplots_adjust(left=0.27, right=0.97, top=0.78, bottom=0.18)

    out_png = HERE / "figures" / "graphical_abstract.png"
    fig.savefig(out_png, dpi=100, bbox_inches=None,
                facecolor="white", edgecolor="none")
    plt.close(fig)

    # Clamp to exactly 920 x 300 px per the JHP graphical-abstract spec.
    from PIL import Image
    with Image.open(out_png) as im:
        if im.size != (920, 300):
            im = im.resize((920, 300), Image.LANCZOS)
            im.save(out_png, optimize=True)
        w, h = im.size

    import os
    size_kb = os.path.getsize(out_png) / 1024.0
    print(f"saved {out_png.name}  {w}x{h} px, {size_kb:.1f} KB")
    print(f"  JHP target: 920x300 px, <= 150 KB  ->  "
          f"{'PASS' if (w == 920 and h == 300 and size_kb <= 150) else 'CHECK'}")


if __name__ == "__main__":
    main()
