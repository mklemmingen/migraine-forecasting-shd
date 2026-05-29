"""Figure A4 - per-patient base-rate distribution.

Each patient's own fraction of headache days and migraine days, across all their
diary days. The wide between-patient spread (some patients near 0, others well
above the cohort mean) is the variation that a pooled C-statistic absorbs as
discrimination - which is why pooled AUROC overstates within-person ranking
(Figure C2). One marker per patient; the solid tick is the cohort mean.

Usage: python fig_a4_base_rates.py
"""
# §11 compliance: per-patient base-rate distribution.
#   §11.1 cohort mean: patient-cluster bootstrap 95% CI (n_boot=1000) rendered as
#       a horizontal whisker through the mean tick and annotated alongside the mean label
#   §11.3 caption + §11.6 footer + §11.11 self-check this block
#   §11.7 N/A (this is the *cohort* base-rate, not a per-cell metric)
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
        y = i + (rng.random(len(r)) - 0.5) * 0.70
        ax.scatter(r, y, s=20, color=S.TARGET[tgt], alpha=0.6, edgecolor="white", lw=0.4)
        # Patient-cluster bootstrap 95% CI on the cohort mean: resamples whole
        # patients with replacement, since the cohort statistic averages over
        # per-patient rates and between-patient variation is the relevant source.
        boot_rng = np.random.default_rng(42)
        boot = np.empty(1000)
        n = len(r)
        for b in range(1000):
            boot[b] = r[boot_rng.integers(0, n, size=n)].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.plot([r.mean(), r.mean()], [i - 0.32, i + 0.32], color=S.INK, lw=1.6)
        ax.plot([lo, hi], [i, i], color=S.INK, lw=1.0, alpha=0.55)
        ax.plot([lo, lo], [i - 0.10, i + 0.10], color=S.INK, lw=1.0, alpha=0.55)
        ax.plot([hi, hi], [i - 0.10, i + 0.10], color=S.INK, lw=1.0, alpha=0.55)
        ax.text(r.mean(), i + 0.42,
                f"mean {r.mean():.1%} [{lo:.1%}, {hi:.1%}]",
                ha="center", fontsize=8)
        print(f"  {tgt:<9} per-patient rate: min {r.min():.1%} median {np.median(r):.1%} "
              f"max {r.max():.1%} | cohort mean {r.mean():.1%} [95% CI {lo:.1%}, {hi:.1%}]")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["headache", "migraine"])
    ax.set_ylim(-0.7, 1.8)
    ax.set_xlabel("per-patient positive-day rate")
    ax.set_title("Between-patient variation in base rate\n"
                 "Park 2016 SHD, n=63 patients (62 enrolled + 1 disability-sheet only)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_a4_base_rates"))


if __name__ == "__main__":
    main()
