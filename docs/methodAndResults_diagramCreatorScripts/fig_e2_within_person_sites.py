"""Figure E2 - within-person discrimination replicates off-site.

Precision-weighted within-person C-statistic on each held-out recruitment site,
per model and target, against chance (0.5) and the internal CV-out-of-fold
within-person estimate (Figure C2). The off-site values cluster around chance just
as internally - the near-chance per-patient result is reproduced on an independent
clinic, not a single-cohort artefact (docs/external_validation_site.md Section 7).
Reads the latest experiment/5/external_site_summary_*.csv.

Usage: python fig_e2_within_person_sites.py
"""
# §11 compliance:
#   §11.1 CIs: DATA-PENDING Hanley-McNeil whiskers on scatter points
#   §11.2 estimability: DATA-PENDING on-plot denominator annotation
#   §11.3 caption + §11.6 footer + §11.7 EPV migraine + §11.11 self-check
import csv
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiment"
MODELS = ["pooled_lr", "add0_stacked", "add1_tabpfn", "add4_window_mlp"]
MLAB = {"pooled_lr": "pooled LR", "add0_stacked": "XGBoost stack",
        "add1_tabpfn": "TabPFN", "add4_window_mlp": "window-MLP"}
SITES = ["uijeongbu", "dongtan"]
# internal CV-OOF within-person (TabPFN, Figure C2 / Addition 5 Section 9b)
INTERNAL = {"headache": 0.542, "migraine": 0.563}


def _latest():
    return max((EXP / "5").glob("external_site_summary_*.csv"), key=lambda p: p.stat().st_mtime)


def main():
    S.apply()
    rows = list(csv.DictReader(open(_latest())))
    d = {(r["target"], r["held_out"], r["model"]): r for r in rows}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.3), sharey=True)
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    for ax, tgt in zip(axes, ("headache", "migraine")):
        for si, site in enumerate(SITES):
            for m in MODELS:
                r = d.get((tgt, site, m))
                if not r or r["within_cstat"] in ("", "nan"):
                    continue
                v = float(r["within_cstat"])
                k = int(r.get("within_k", 0))
                lo = float(r.get("within_ci_low", "nan")) if r.get("within_ci_low") not in (None, "") else float("nan")
                hi = float(r.get("within_ci_high", "nan")) if r.get("within_ci_high") not in (None, "") else float("nan")
                xj = si + (rng.random() - 0.5) * 0.55
                if lo == lo and hi == hi:
                    ax.errorbar(xj, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]],
                                fmt="o", ms=8, color=S.ARCH[m], alpha=0.85,
                                mec="white", mew=0.5, ecolor=S.SOFT, capsize=0,
                                label=MLAB[m] if si == 0 else None)
                else:
                    ax.scatter(xj, v, s=55, color=S.ARCH[m], alpha=0.85,
                               edgecolor="white", lw=0.5,
                               label=MLAB[m] if si == 0 else None)
                print(f"  {tgt:<9} {site:<10} {m:<16} within {v:.3f} "
                      f"[{lo:.3f}-{hi:.3f}] (k={k})")
        S.refline(ax, y=0.5, label="chance")
        # neutral reference line + per-panel value, so the shared legend swatch
        # cannot mismatch the drawn colour (the internal estimate is target-specific)
        ax.axhline(INTERNAL[tgt], color=S.SOFT, lw=1.4)
        ax.text(0.02, INTERNAL[tgt], f"internal C {INTERNAL[tgt]:.2f}",
                transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color=S.SOFT)
        # Add estimability denominator annotation per panel (bottom-left, away from legend).
        ks = [int(r.get("within_k", 0)) for s in SITES for m in MODELS
              if (r := d.get((tgt, s, m))) is not None]
        if ks:
            kmax = max(ks)
            kmin = min(ks)
            klabel = f"k = {kmin} estimable" if kmin == kmax else f"k = {kmin}-{kmax} estimable per site"
            ax.text(0.02, 0.04, klabel, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=7, color=S.GREY, style="italic")
        ax.set_xticks(range(len(SITES)))
        ax.set_xticklabels([f"held-out\n{s}" for s in SITES])
        ax.set_ylim(0.40, 0.85)
        ax.set_title(tgt)
        S.epv_annotation(ax, tgt, cell="full_features", loc="upper right")
    axes[0].set_ylabel("within-person C-statistic")
    axes[0].legend(fontsize=8, loc="upper center", ncol=2)
    fig.suptitle("Within-person discrimination stays near chance off-site\n"
                 "Park 2016 SHD, n=62; leave-one-site-out", y=1.04, fontsize=10)
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_e2_within_person_sites"))


if __name__ == "__main__":
    main()
