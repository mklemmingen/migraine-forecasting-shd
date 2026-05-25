"""Contact sheet of every paper figure - input for the cross-figure consistency
review (Tier 1b). One image lets a reviewer check the SET at once: same entity ->
same colour everywhere, uniform fonts/legends, comparable sizing - the consistency
a per-figure review structurally cannot see.

Usage: python _montage.py   ->  figures/_montage.png
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"


def main():
    pngs = sorted(p for p in FIGDIR.glob("fig_*.png"))
    n = len(pngs)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p in zip(axes.ravel(), pngs):
        ax.imshow(mpimg.imread(p))
        ax.set_title(p.stem, fontsize=8)
    out = FIGDIR / "_montage.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"montage of {n} figures -> {out}")


if __name__ == "__main__":
    main()
