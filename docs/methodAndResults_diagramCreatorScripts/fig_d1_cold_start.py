"""Figure D1 - cold-start curve.

Brier score of two causal forecasters by how many of a patient's own diary days
precede the predicted day: the fixed cohort rate (population) versus the patient's
empirical-Bayes running attack rate (personalised). Personalisation overtakes the
population baseline after only a handful of own days - the gain is a base-rate
level effect, not within-person day-to-day ranking (Addition 5 Section 9e).
Row-weighted per own-day band. Reuses Addition 5's walkforward.cold_start_curve.

Usage: python fig_d1_cold_start.py
"""
# §11 compliance:
#   §11.1 PASS: patient-cluster bootstrap 95% CI envelope per curve (population +
#               personalised); n_boot=500
#   §11.3 caption: cohort+n in suptitle; §11.6 footer; §11.7 EPV migraine; §11.11 self-check
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path.insert(0, str(EXP / "5" / "_personal"))
import walkforward as WF  # noqa: E402


def main():
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.2))
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    for ax, tgt in zip(axes, ("headache", "migraine")):
        diary = pd.read_parquet(REPO / "data" / "processed" / tgt / "diary_cv5_timeseries.parquet")
        b = WF.binned_curve_ci(diary, "migraine_target", alpha=5.0)
        cp = WF.cold_start_point(WF.cold_start_curve(diary, "migraine_target", alpha=5.0))
        x = list(range(len(b)))
        ax.fill_between(x, b["pop_ci_low"], b["pop_ci_high"],
                        color=S.GREY, alpha=0.15, linewidth=0)
        ax.fill_between(x, b["pers_ci_low"], b["pers_ci_high"],
                        color=S.TARGET[tgt], alpha=0.15, linewidth=0)
        ax.plot(x, b["brier_population"], "o--", color=S.GREY,
                label="population (cohort rate)")
        ax.plot(x, b["brier_personalised"], "o-", color=S.TARGET[tgt],
                label="personalised (own running rate)")
        ax.set_xticks(list(x)); ax.set_xticklabels(b["own_days"])
        ax.set_xlabel("own diary days seen")
        ax.set_ylabel("Brier score (lower is better)")
        ax.set_title(tgt)
        ax.legend(fontsize=8)
        S.epv_annotation(ax, tgt, cell="full_features", loc="upper right")
        print(f"  {tgt}: cold-start point n_prior={cp}")
    fig.suptitle("Cold start: own running rate overtakes the cohort rate within days\n"
                 "Park 2016 SHD, n=62; patient-cluster bootstrap 95% CI envelope",
                 y=1.04, fontsize=10)
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_d1_cold_start"))


if __name__ == "__main__":
    main()
