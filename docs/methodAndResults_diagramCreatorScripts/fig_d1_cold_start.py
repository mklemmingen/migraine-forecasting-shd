"""Figure D1 - cold-start curve.

Brier score of two causal forecasters by how many of a patient's own diary days
precede the predicted day: the fixed cohort rate (population) versus the patient's
empirical-Bayes running attack rate (personalised). Personalisation overtakes the
population baseline after only a handful of own days - the gain is a base-rate
level effect, not within-person day-to-day ranking (Addition 5 Section 9e).
Row-weighted per own-day band. Reuses Addition 5's walkforward.cold_start_curve.

Usage: python fig_d1_cold_start.py
"""
import sys
from pathlib import Path

import _figstyle as S
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path.insert(0, str(EXP / "5" / "_personal"))
import walkforward as WF  # noqa: E402


def main():
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, tgt in zip(axes, ("headache", "migraine")):
        diary = pd.read_parquet(REPO / "data" / "processed" / tgt / "diary_cv5_timeseries.parquet")
        curve = WF.cold_start_curve(diary, "migraine_target", alpha=5.0)
        b = WF.binned_curve(curve)
        cp = WF.cold_start_point(curve)
        x = range(len(b))
        ax.plot(x, b["brier_population"], "o--", color="#999999",
                label="population (cohort rate)")
        ax.plot(x, b["brier_personalised"], "o-", color=S.TARGET[tgt],
                label="personalised (own running rate)")
        ax.set_xticks(list(x)); ax.set_xticklabels(b["own_days"])
        ax.set_xlabel("own diary days seen")
        ax.set_ylabel("Brier score (lower is better)")
        ax.set_title(f"{tgt}  (personalisation helps from ~{cp} own day{'s' if cp != 1 else ''})")
        ax.legend(fontsize=8)
        print(f"  {tgt}: cold-start point n_prior={cp}")
        print(b.to_string(index=False))
    fig.suptitle("Cold start: own running rate overtakes the cohort rate within days",
                 y=1.02, fontsize=12)
    print("saved", S.save(fig, HERE / "figures" / "fig_d1_cold_start"))


if __name__ == "__main__":
    main()
