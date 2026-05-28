"""Figure A3 - diary-coverage calendar heatmap.

Patient (rows, sorted by enrolment start) by calendar date (columns), coloured by
the day's record: absent, headache-free, headache (non-migraine), or migraine.
Shows the staggered enrolment and coverage gaps - the reason a late-enrolment
chronological hold-out is sparse per patient and the within-person estimate needs
the CV out-of-fold pooling (Figure C2).

Usage: python fig_a3_coverage.py
"""
# §11 compliance: coverage heatmap (no per-cell metric).
#   §11.3 caption: cohort+n in title; §11.6 footer; §11.11 self-check this block
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "data" / "processed"


def main():
    S.apply()
    mig = pd.read_parquet(BASE / "migraine/diary_cv5_timeseries.parquet")[
        ["patient_id", "date", "migraine_target"]].rename(columns={"migraine_target": "mig"})
    hea = pd.read_parquet(BASE / "headache/diary_cv5_timeseries.parquet")[
        ["patient_id", "date", "migraine_target"]].rename(columns={"migraine_target": "hea"})
    df = mig.merge(hea, on=["patient_id", "date"])
    df["date"] = pd.to_datetime(df["date"])
    # day code: 1 present/headache-free, 2 headache (non-migraine), 3 migraine
    df["code"] = 1 + (df["hea"] > 0).astype(int) + (df["mig"] > 0).astype(int)

    order = df.groupby("patient_id")["date"].min().sort_values().index
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    grid = np.zeros((len(order), len(dates)))            # 0 = absent
    di = {d: j for j, d in enumerate(dates)}
    pi = {p: i for i, p in enumerate(order)}
    for p, d, c in zip(df["patient_id"], df["date"], df["code"]):
        grid[pi[p], di[d]] = c

    cmap = ListedColormap(S.COVERAGE)        # absent, headache-free, headache, migraine
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=S.figsize("double", 5.2))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    # Thin white minor gridlines on every cell boundary for cell-level
    # readability; spec rule for discrete heatmaps.
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.2)
    ax.tick_params(which="minor", length=0)
    # month ticks
    months = pd.date_range(dates.min(), dates.max(), freq="MS")
    ax.set_xticks([di[m] for m in months if m in di])
    ax.set_xticklabels([m.strftime("%b %Y") for m in months if m in di], rotation=45, ha="right")
    ax.set_ylabel("patient (sorted by enrolment start)")
    ax.set_xlabel("calendar date")
    coverage_pct = 100 * (grid > 0).mean()
    ax.set_title(f"Diary coverage: 63 patients, {coverage_pct:.1f}% coverage "
                 f"(staggered enrolment, gaps shown as white)")
    from matplotlib.patches import Patch
    S.framed_legend(ax, handles=[Patch(fc="white", ec="#bbbbbb", label="no entry"),
                                 Patch(fc=S.COVERAGE[1], label="headache-free"),
                                 Patch(fc=S.COVERAGE[2], label="headache"),
                                 Patch(fc=S.COVERAGE[3], label="migraine")],
                    loc="lower right")
    print(f"  grid {grid.shape[0]} patients x {grid.shape[1]} days; "
          f"coverage {coverage_pct:.1f}%")
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_a3_coverage"))


if __name__ == "__main__":
    main()
