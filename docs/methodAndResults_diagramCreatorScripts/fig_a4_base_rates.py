"""Figure A4 - per-patient base-rate distribution.

Each patient's own fraction of headache days and migraine days, across all their
diary days. The wide between-patient spread (some patients near 0, others well
above the cohort mean) is the variation that a pooled C-statistic absorbs as
discrimination - which is why pooled AUROC overstates within-person ranking
(Figure C2). One marker per patient; the solid tick is the cohort mean.

Usage: python fig_a4_base_rates.py
"""
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "data" / "processed"


def _rates(target):
    df = pd.read_parquet(BASE / target / "diary_cv5_timeseries.parquet")
    return df.groupby("patient_id")["migraine_target"].mean()


def main():
    S.apply()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=S.figsize("double", 4.0))
    for i, tgt in enumerate(("headache", "migraine")):
        r = _rates(tgt).to_numpy()
        y = i + (rng.random(len(r)) - 0.5) * 0.5
        ax.scatter(r, y, s=22, color=S.TARGET[tgt], alpha=0.6, edgecolor="white", lw=0.4)
        ax.plot([r.mean(), r.mean()], [i - 0.32, i + 0.32], color=S.INK, lw=1.6)
        ax.text(r.mean(), i + 0.42, f"mean {r.mean():.1%}", ha="center", fontsize=8)
        print(f"  {tgt:<9} per-patient rate: min {r.min():.1%} median {np.median(r):.1%} "
              f"max {r.max():.1%} | cohort mean {r.mean():.1%}")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["headache", "migraine"])
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlabel("per-patient positive-day rate")
    ax.set_title("Between-patient variation in base rate (n=63)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    print("saved", S.save(fig, HERE / "figures" / "fig_a4_base_rates"))


if __name__ == "__main__":
    main()
