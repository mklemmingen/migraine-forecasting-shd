"""Figure 1 (working-notes ID: A2) - cohort flow diagram (TRIPOD-style participant flow).

From the published diary export to the headline evaluation split, with the
site-reconciliation exclusion noted. Counts are computed from the processed
parquet files, not transcribed (see the print-out). The two targets share the
same patient-days and split (only the day-label differs), so the split sizes are
identical across targets.

Usage: python fig_a2_cohort_flow.py
"""
# §11 compliance: cohort-flow schematic (no headline metric).
#   §11.3 self-contained caption:        title carries cohort name + n
#   §11.6 CC BY 4.0 footer:              S.cc_by_footer() invoked
#   §11.1, §11.2, §11.5, §11.7:          N/A
#   §11.10 no "substantial"/"large":     verified
#   §11.11 self-check:                   this block
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

    fig, ax = plt.subplots(figsize=S.figsize("double", 9.0))
    # Limits hug the funnel (lowest box bottom 1.2, top box top 14.1) so
    # bbox="tight" leaves no dead band below the final split box. The right edge
    # runs to 10.4 (not 10.0) so the exclusion boxes at x=9..10 keep their right
    # border inside the axes clip region instead of being shaved off.
    ax.set_xlim(0, 10.4); ax.set_ylim(1.0, 14.4); ax.axis("off")

    # Park 2016 upstream enrolment funnel transcribed from the PLOS ONE article
    # page 4 prose paragraph: "Initially, 113 patients were recruited from two
    # centers. However, 30 patients withdrew before the end of the study;
    # therefore, 83 patients finished the study. Of these, 62 patients kept a
    # diary for at least 50% of the study period."
    bup1 = S.box(ax, (5, 13.7), 7.2, 0.85,
                 "113 episodic-migraine patients recruited at two Korean neurology clinics\n"
                 "(Park 2016 PLOS ONE 11(2):e0149577, page 4)", fc=S.PALE_FILL)
    bex_withdraw = S.box(ax, (9.0, 12.9), 2.0, 0.6,
                         "- 30 withdrew\nbefore end of study",
                         role="exclude", fontsize=7.5)
    bup2 = S.box(ax, (5, 12.1), 7.2, 0.85,
                 "83 patients finished the 3-month diary study", fc=S.PALE_FILL)
    bex_adherence = S.box(ax, (9.0, 11.3), 2.0, 0.6,
                          "- 21 with diary\nadherence < 50%",
                          role="exclude", fontsize=7.5)
    bup3 = S.box(ax, (5, 10.5), 7.2, 0.85,
                 "62 patients kept diary >= 50% of period\n"
                 "(Park 2016 analytic cohort)", fc=S.PALE_FILL)

    b1 = S.box(ax, (5, 9.1), 7.2, 1.3,
               f"SHD smartphone headache diary - Park et al. 2016 (CC BY)\n"
               f"{n_pt} patient_ids (62 Park analytic + 1 reconciliation),"
               f" {n_rows:,} patient-days\n"
               f"{d0:%Y-%m-%d} to {d1:%Y-%m-%d}",
               fc=S.PALE_FILL, weight="bold", fontsize=8)
    b2 = S.box(ax, (5, 6.8), 7.2, 1.3,
               "Recruitment site recovered from Sheet 1\n"
               "62 patients mapped: Uijeongbu 32 - Dongtan 30")
    bex = S.box(ax, (9.0, 7.95), 2.0, 0.85,
                "- 1 reconciliation pt\n(- 38 days), excluded\nfrom site analysis",
                role="exclude", fontsize=7.5)
    bh = S.box(ax, (2.7, 4.6), 3.4, 1.15, f"Headache target\n{hea_r:.1%} of all days",
               fc=S.PALE_FILL, ec=S.TARGET["headache"])
    bmg = S.box(ax, (7.3, 4.6), 3.4, 1.15, f"Migraine target\n{mig_r:.1%} of all days",
                fc=S.PALE_FILL, ec=S.TARGET["migraine"])
    b4 = S.box(ax, (5, 2.0), 7.6, 1.6,
               f"Chronological 70/15/15 split (illustrative; A5 shows stratified / patient / site)\n"
               f"train {sizes['train']:,}  -  val {sizes['val']:,}  -  test {sizes['test']:,}\n"
               f"identical row partitioning across both targets",
               role="output")
    # Upstream-funnel arrows
    S.arrow(ax, (bup1[0], bup1[1] - bup1[3] / 2), (bup2[0], bup2[1] + bup2[3] / 2))
    S.arrow(ax, (5.0, 12.9), (bex_withdraw[0] - bex_withdraw[2] / 2, 12.9))
    S.arrow(ax, (bup2[0], bup2[1] - bup2[3] / 2), (bup3[0], bup3[1] + bup3[3] / 2))
    S.arrow(ax, (5.0, 11.3), (bex_adherence[0] - bex_adherence[2] / 2, 11.3))
    S.arrow(ax, (bup3[0], bup3[1] - bup3[3] / 2), (b1[0], b1[1] + b1[3] / 2))
    # Pipeline arrows (unchanged)
    for a, b in [(b1, b2)]:
        S.arrow(ax, (a[0], a[1] - a[3] / 2), (b[0], b[1] + b[3] / 2))
    S.arrow(ax, (5.0, 7.95), (bex[0] - bex[2] / 2, 7.95))   # branch to the exclusion note
    S.arrow(ax, (b2[0], b2[1] - b2[3] / 2), (bh[0], bh[1] + bh[3] / 2))
    S.arrow(ax, (b2[0], b2[1] - b2[3] / 2), (bmg[0], bmg[1] + bmg[3] / 2))
    S.arrow(ax, (bh[0], bh[1] - bh[3] / 2), (b4[0] - 1.5, b4[1] + b4[3] / 2))
    S.arrow(ax, (bmg[0], bmg[1] - bmg[3] / 2), (b4[0] + 1.5, b4[1] + b4[3] / 2))
    ax.set_title("Cohort flow - Park 2016 SHD: enrolment funnel (113 recruited -> 62 analytic)\n"
                 f"and processed pipeline ({n_pt} patient_ids, {n_rows:,} patient-days)",
                 fontsize=11.5)
    print(f"  Park funnel: 113 recruited -> 83 finished (-30 withdrew) -> 62 >=50% adherence (-21)")
    print(f"  Processed: {n_pt} patient_ids (62 Park + 1 reconciliation), {n_rows} days; "
          f"headache {hea_r:.1%}, migraine {mig_r:.1%}; split {sizes}")
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_a2_cohort_flow"))


if __name__ == "__main__":
    main()
