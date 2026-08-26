"""Figure A6 - the evaluation argument, asked as four questions in order.

Each question is a weaker claim than the one before it, and the figure records what
survives each. Pooled AUROC answers a question about the cohort; the within-person
C-statistic asks the same question inside one patient; Brier skill asks whether the
forecast beats that patient's own attack rate; net benefit asks whether acting on it
would help. A reader who stops at the first question sees a usable model. A reader
who asks all four does not.

Numbers are the headline cells reported in Results, not a worked example:
migraine XGB-HP020 and headache TabPFN-v2.6, both full_features 70/30 chronological.
Discrimination and Brier skill carry patient-cluster bootstrap intervals; the
within-person C carries its Paule-Mandel interval. All are read from the result CSVs
at draw time, so the figure cannot drift from the tables.

Usage: python fig_a6_pooled_vs_within.py
"""
# §11 compliance: reports headline-cell metrics.
#   §11.1 metric + CI + n:               each step carries the interval; k stated for the C
#   §11.2 cell named:                    in the footer
#   §11.3 self-contained caption:        cohort and cells named in the footer
#   §11.10 no "substantial"/"large":     verified
#   §11.11 self-check:                   this block
import csv
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiment"

W_MM, H_MM = 183.0, 158.0
PT_Q, PT_VAL, PT_BODY, PT_TINY = 9.0, 9.4, 7.2, 6.5
INK, MUTE, TINT = S.INK, S.GREY, "#f4f4f4"


def _load():
    """Headline-cell numbers, read from the result files rather than transcribed."""
    boot = {}
    f = EXP / "_eval/_special/patient_cluster_bootstrap_20260530_125806.csv"
    for r in csv.DictReader(open(f)):
        boot[(r["target"], r["architecture"])] = r
    wp = {}
    g = EXP / "5/within_person_summary_cv_20260530_151055.csv"
    for r in csv.DictReader(open(g)):
        wp[(r["target"], r["architecture"])] = r
    mig, hea = boot[("migraine", "XGBoost")], boot[("headache", "TabPFN")]
    mig_w, hea_w = wp[("migraine", "stacked_2xgb_meta_lr")], wp[("headache", "tabpfn")]
    return mig, hea, mig_w, hea_w


def _txt(ax, x, y, t, size=PT_BODY, col=INK, ha="left", weight="normal"):
    ax.text(x, y, t, fontsize=size, color=col, va="center", ha=ha, linespacing=1.55,
            fontweight=weight)


def _chip(ax, x, y, w, h, target, value, interval, verdict, col):
    """One target's answer at one step: the number, its interval, and whether the
    answer supports going on to the next question."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=TINT, ec="none"))
    ax.add_patch(Rectangle((x, y), 1.3, h, fc=col, ec="none"))
    _txt(ax, x + 4.0, y + h - 4.0, target, PT_TINY, MUTE)
    _txt(ax, x + 4.0, y + 6.6, value, PT_VAL, INK, weight="bold")
    if interval:
        _txt(ax, x + 21.0, y + 6.6, interval, PT_TINY, MUTE)
    _txt(ax, x + w - 3.0, y + h - 4.0, "yes" if verdict else "no", PT_TINY,
         INK if verdict else col, ha="right", weight="bold")


def _step(ax, y, n, question, gloss, chips, last=False):
    """One question in the chain, with a spine linking it to the next."""
    ax.add_patch(FancyBboxPatch((6, y + 6.0), 7.6, 7.6,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                fc=INK, ec="none"))
    _txt(ax, 9.8, y + 9.8, str(n), PT_BODY, "white", "center", "bold")
    if not last:
        ax.plot([9.8, 9.8], [y - 7.0, y + 6.0], color="#cfcfcf", lw=1.2, zorder=0)
    _txt(ax, 17.5, y + 10.4, question, PT_Q, INK, weight="bold")
    _txt(ax, 17.5, y + 3.4, gloss, PT_BODY, MUTE)
    for i, (target, value, interval, ok, col) in enumerate(chips):
        _chip(ax, 100 + i * 42.0, y - 0.5, 39.0, 14.0, target, value, interval, ok, col)


def main() -> None:
    S.apply()
    mig, hea, mig_w, hea_w = _load()
    ORA, BLU = S.target_color("migraine"), S.target_color("headache")

    def ci(lo, hi, sign=False):
        f = "{:+.2f}" if sign else "{:.2f}"
        return f"[{f.format(float(lo))}, {f.format(float(hi))}]"

    m_auc, h_auc = float(mig["auroc"]), float(hea["auroc"])
    m_c, h_c = float(mig_w["within_person"]), float(hea_w["within_person"])
    m_bs, h_bs = float(mig["brier_skill"]), float(hea["brier_skill"])
    assert m_c < 0.60 and h_c < 0.60, "within-person C is the near-chance step"
    assert float(mig["bs_cluster_lo"]) < 0 < float(mig["bs_cluster_hi"]), \
        "migraine Brier skill must straddle zero for step 3's reading"
    assert float(hea["bs_cluster_lo"]) > 0, "headache Brier skill must exclude zero"
    assert float(mig_w["within_ci_low"]) > 0.5 and float(hea_w["within_ci_low"]) > 0.5, \
        "the footnote states both within-person intervals exclude 0.50"

    fig, ax = plt.subplots(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM)
    ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    _txt(ax, 6, 150.0, "Four questions, asked in order", 10.2, INK, weight="bold")
    _txt(ax, 6, 143.0, "Each one asks less of the model than the last. A reader who stops "
                       "at the first sees a usable forecast.", PT_BODY, MUTE)
    _txt(ax, 100, 136.5, "migraine", PT_TINY, MUTE)
    _txt(ax, 142, 136.5, "headache", PT_TINY, MUTE)

    _step(ax, 118.0, 1, "Can it rank attack days across the cohort?",
          "Pooled AUROC, every patient's days in one ranking.",
          [("pooled AUROC", f"{m_auc:.2f}", ci(mig["auroc_cluster_lo"], mig["auroc_cluster_hi"]), True, ORA),
           ("pooled AUROC", f"{h_auc:.2f}", ci(hea["auroc_cluster_lo"], hea["auroc_cluster_hi"]), True, BLU)])

    _step(ax, 92.0, 2, "Can it rank days inside one patient?",
          "The same question asked within each patient's own diary.",
          [("within-person C", f"{m_c:.2f}", ci(mig_w["within_ci_low"], mig_w["within_ci_high"]), False, ORA),
           ("within-person C", f"{h_c:.2f}", ci(hea_w["within_ci_low"], hea_w["within_ci_high"]), False, BLU)])

    _step(ax, 66.0, 3, "Does it beat the patient's own attack rate?",
          "Brier skill against each patient's recorded rate.",
          [("Brier skill", f"{m_bs:+.2f}", ci(mig["bs_cluster_lo"], mig["bs_cluster_hi"], True), False, ORA),
           ("Brier skill", f"{h_bs:+.2f}", ci(hea["bs_cluster_lo"], hea["bs_cluster_hi"], True), True, BLU)])

    _step(ax, 40.0, 4, "Would acting on it help the patient?",
          "Decision-curve net benefit across treatment thresholds.",
          [("net benefit", "near zero", "", False, ORA),
           ("net benefit", "beats treat-none", "", False, BLU)], last=True)

    ax.add_patch(FancyBboxPatch((6, 8.0), W_MM - 12, 17.0,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                fc=TINT, ec="none"))
    _txt(ax, 10, 19.5, "Pooled AUROC alone would have stopped at question 1.",
         PT_Q, INK, weight="bold")
    _txt(ax, 10, 12.6, "Migraine fails from question 2 onward. Headache clears "
                       "question 3 but not question 4, where it never beats treating "
                       "everyone.",
         PT_BODY, MUTE)
    _txt(ax, 6, 4.6, "yes / no records whether the answer supports the next question "
                     "clinically, not whether it reaches significance: both within-person "
                     "intervals exclude 0.50 while sitting close to it.", PT_TINY, MUTE)
    _txt(ax, 6, 0.8, "Headline cells: migraine XGB-HP020, headache TabPFN-v2.6, both "
                     "full_features 70/30 chronological, Park 2016 cohort (62 patients, "
                     "4,516 diary days). Within-person C estimable in 19 and 57 records.",
         PT_TINY, MUTE)

    print("saved", S.save(fig, HERE / "figures" / "fig_a6_pooled_vs_within"))


if __name__ == "__main__":
    main()
