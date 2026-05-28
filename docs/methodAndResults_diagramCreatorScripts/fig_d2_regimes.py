"""Figure D2 - personalisation-regime comparison.

Pooled AUROC and the precision-weighted within-person C-statistic for the three
personalisation regimes (pooled global LR, per-patient LR, partial-pooling
empirical-Bayes intercept), per target, via CV out-of-fold. The pooled-vs-within
gap is the load-bearing message: pooled lift exists (small on CV-OOF, larger on
held-out per Add 5 Section 9d) but the within-person C-statistic stays flat near
chance, so any pooled gain is between-patient base-rate separation, not
within-person ranking. Replicates the regime CV-OOF loop inline.

Usage: python fig_d2_regimes.py
"""
# §11 compliance:
#   §11.1 partial PASS (regime within-person C has error bars); pooled bars: DATA-PENDING bootstrap
#   §11.2 add estimability denominator on plot; §11.3 caption cohort+n
#   §11.6 footer; §11.7 EPV migraine; §11.11 self-check this block
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path.insert(0, str(EXP / "5" / "_personal"))
sys.path.insert(0, str(EXP))
import regimes as RG  # noqa: E402
import within_person as WP  # noqa: E402
from _dataRead.read import load_raw, NON_FEATURE_COLS, TARGET_COL  # noqa: E402

FEATURE_SET = "full_features"
REGIMES = ["pooled", "per_patient", "partial_pool"]
RLAB = {"pooled": "pooled\nglobal LR", "per_patient": "per-patient\nLR",
        "partial_pool": "partial\npooling (EB)"}


def _regime_oof(name, cv, fc, n_splits=5):
    fn = RG.REGIMES[name]
    ys, ps, pids = [], [], []
    for fold in range(1, n_splits + 1):
        tr, ev = cv[cv["cv_fold"] < fold], cv[cv["cv_fold"] == fold]
        if tr.empty or ev.empty:
            continue
        _, p = fn(tr, ev, ev, fc)
        ps.append(np.asarray(p, float)); ys.append(ev[TARGET_COL].to_numpy(float))
        pids.append(ev["patient_id"].astype(str).to_numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(pids)


def main():
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.3), sharey=True)
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    for ax, tgt in zip(axes, ("headache", "migraine")):
        cv = load_raw(str(REPO / "data" / "processed" / tgt / "diary_cv5_timeseries.parquet"))
        fc = [c for c in cv.columns if c not in NON_FEATURE_COLS]
        pooled_auc, within, ci_lo, ci_hi = [], [], [], []
        for name in REGIMES:
            y, p, pid = _regime_oof(name, cv, fc)
            w = WP.within_person_cstatistic(WP.per_patient_scores(y, p, pid, WP.MIN_POS))
            pa = float(roc_auc_score(y, p))
            pooled_auc.append(pa); within.append(w["estimate"])
            ci_lo.append(w["estimate"] - w["ci_low"]); ci_hi.append(w["ci_high"] - w["estimate"])
            print(f"  {tgt:<9} {name:<13} pooled {pa:.3f} | within {w['estimate']:.3f} "
                  f"[{w['ci_low']:.3f}-{w['ci_high']:.3f}]")
        x = np.arange(len(REGIMES))
        ax.plot(x, pooled_auc, "o-", color=S.TARGET[tgt], lw=1.6, label="pooled AUROC")
        ax.errorbar(x, within, yerr=[ci_lo, ci_hi], fmt="s--", color=S.SOFT, lw=1.3,
                    capsize=3, label="within-person C")
        S.refline(ax, y=0.5, ls=":", label="chance")
        ax.set_xticks(x); ax.set_xticklabels([RLAB[r] for r in REGIMES])
        ax.set_ylim(0.40, 0.85)
        ax.set_title(tgt)
    axes[0].set_ylabel("AUROC / C-statistic")
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle(f"Personalisation regimes ({FEATURE_SET}): pooled gain is between-patient\n"
                 "Park 2016 SHD, n=62; within-person C with Hanley-McNeil 95% CI",
                 y=1.02)
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_d2_regimes"))


if __name__ == "__main__":
    main()
