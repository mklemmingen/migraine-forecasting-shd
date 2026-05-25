"""Figure A2 - cohort flow diagram (TRIPOD-style participant flow).

From the published diary export to the headline evaluation split, with the
site-reconciliation exclusion noted. Counts are computed from the processed
parquet files, not transcribed (see the print-out). The two targets share the
same patient-days and split (only the day-label differs), so the split sizes are
identical across targets.

Usage: python fig_a2_cohort_flow.py
"""
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "data" / "processed"


def main():
    S.apply()
    mig = pd.read_parquet(BASE / "migraine/diary_cv5_timeseries.parquet")
    hea = pd.read_parquet(BASE / "headache/diary_cv5_timeseries.parquet")
    n_pt, n_rows = mig["patient_id"].nunique(), len(mig)
    d0, d1 = pd.to_datetime(mig["date"]).min(), pd.to_datetime(mig["date"]).max()
    mig_r, hea_r = mig["migraine_target"].mean(), hea["migraine_target"].mean()
    sizes = {sp: len(pd.read_parquet(BASE / f"migraine/70_15_15/chrono/diary_{sp}.parquet"))
             for sp in ("train", "val", "test")}

    fig, ax = plt.subplots(figsize=S.figsize("double", 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    b1 = S.box(ax, (5, 9.1), 7.2, 1.3,
               f"SHD smartphone headache diary - Park et al. 2016 (CC BY)\n"
               f"{n_pt} patients, {n_rows:,} patient-days\n"
               f"{d0:%Y-%m-%d} to {d1:%Y-%m-%d}", fc=S.PALE_FILL, weight="bold")
    b2 = S.box(ax, (5, 6.8), 7.2, 1.3,
               "Recruitment site recovered from Sheet 1\n"
               "62 patients mapped: Uijeongbu 32 - Dongtan 30")
    bex = S.box(ax, (8.5, 7.95), 2.8, 0.85,
                "- 1 reconciliation patient\n(38 days), excluded\nfrom site analysis",
                role="exclude", fontsize=7.5)
    bh = S.box(ax, (2.7, 4.6), 3.4, 1.15, f"Headache target\n{hea_r:.1%} of all days",
               fc=S.PALE_FILL, ec=S.TARGET["headache"])
    bmg = S.box(ax, (7.3, 4.6), 3.4, 1.15, f"Migraine target\n{mig_r:.1%} of all days",
                fc=S.PALE_FILL, ec=S.TARGET["migraine"])
    b4 = S.box(ax, (5, 2.2), 7.6, 1.35,
               f"Chronological 70/15/15 split (identical rows across targets)\n"
               f"train {sizes['train']:,}  -  val {sizes['val']:,}  -  test {sizes['test']:,}",
               role="output")
    for a, b in [(b1, b2)]:
        S.arrow(ax, (a[0], a[1] - a[3] / 2), (b[0], b[1] + b[3] / 2))
    S.arrow(ax, (5.0, 7.95), (bex[0] - bex[2] / 2, 7.95))   # branch to the exclusion note
    S.arrow(ax, (b2[0], b2[1] - b2[3] / 2), (bh[0], bh[1] + bh[3] / 2))
    S.arrow(ax, (b2[0], b2[1] - b2[3] / 2), (bmg[0], bmg[1] + bmg[3] / 2))
    S.arrow(ax, (bh[0], bh[1] - bh[3] / 2), (b4[0] - 1.5, b4[1] + b4[3] / 2))
    S.arrow(ax, (bmg[0], bmg[1] - bmg[3] / 2), (b4[0] + 1.5, b4[1] + b4[3] / 2))
    ax.set_title("Cohort flow")
    print(f"  {n_pt} patients, {n_rows} days; headache {hea_r:.1%}, migraine {mig_r:.1%}; "
          f"split {sizes}")
    print("saved", S.save(fig, HERE / "figures" / "fig_a2_cohort_flow"))


if __name__ == "__main__":
    main()
