"""Figure E1 - calibration drift across sites (leave-one-site-out).

Observed-to-expected ratio (O:E = observed rate / mean predicted) on each held-out
recruitment site, per model and target. A model carried to a lower-base-rate site
over-predicts (O:E < 1); carried to a higher-base-rate site it under-predicts
(O:E > 1). The drift tracks the base-rate gap rather than the model, the
distinctive transportability result (docs/external_validation_site.md Section 7).
Reads the latest experiment/5/external_site_summary_*.csv (regenerate with
`python experiment/5/run_external_site.py --no-emit`).

Usage: python fig_e1_calibration_drift.py
"""
# §11 compliance:
#   §11.1 CIs: yerr whiskers from oe_ratio_ci_low/high in external_site_summary csv
#   §11.5 per-direction: numeric per-direction values (0.50/1.54) in suptitle
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


def _latest():
    return max((EXP / "5").glob("external_site_summary_*.csv"), key=lambda p: p.stat().st_mtime)


def main():
    S.apply()
    rows = list(csv.DictReader(open(_latest())))
    d = {(r["target"], r["held_out"], r["model"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.3), sharey=True)
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    w = 0.2
    for ax, tgt in zip(axes, ("headache", "migraine")):
        xticklab = []
        for si, site in enumerate(SITES):
            any_r = next((d[(tgt, site, m)] for m in MODELS if (tgt, site, m) in d), None)
            if any_r is None:
                xticklab.append(site); continue
            obs, trn = float(any_r["observed_rate"]), float(any_r["train_rate"])
            xticklab.append(f"{site}\ntest {obs:.1%} / train {trn:.1%}")
            for mi, m in enumerate(MODELS):
                r = d.get((tgt, site, m))
                if not r:
                    continue
                oe = float(r["oe_ratio"])
                oe_lo = float(r.get("oe_ratio_ci_low", "nan")) if r.get("oe_ratio_ci_low") not in (None, "") else float("nan")
                oe_hi = float(r.get("oe_ratio_ci_high", "nan")) if r.get("oe_ratio_ci_high") not in (None, "") else float("nan")
                # bars emanate from the O:E=1 neutral, so bar length = deviation
                # from perfect calibration (a ratio drawn from 0 would exaggerate
                # over-prediction relative to equal-magnitude under-prediction).
                yerr_arr = None
                if oe_lo == oe_lo and oe_hi == oe_hi:
                    yerr_arr = [[max(oe - oe_lo, 0)], [max(oe_hi - oe, 0)]]
                ax.bar(si + (mi - 1.5) * w, oe - 1.0, w, bottom=1.0, color=S.ARCH[m],
                       yerr=yerr_arr, capsize=2, ecolor=S.SOFT,
                       label=MLAB[m] if si == 0 else None)
                print(f"  {tgt:<9} {site:<10} {m:<16} O:E {oe:.2f} "
                      f"[{oe_lo:.2f}-{oe_hi:.2f}]")
        S.refline(ax, y=1.0)
        ax.text(0.5, 1.0, "perfect (O:E = 1)", transform=ax.get_yaxis_transform(),
                ha="center", va="bottom", fontsize=7, color=S.INK)
        ax.set_ylim(0.4, 1.65)
        ax.text(0.99, 0.72, "under-predicts", transform=ax.transAxes, ha="right",
                va="center", fontsize=7, color=S.GREY, style="italic")
        ax.text(0.99, 0.18, "over-predicts", transform=ax.transAxes, ha="right",
                va="center", fontsize=7, color=S.GREY, style="italic")
        ax.set_xticks(range(len(SITES))); ax.set_xticklabels(xticklab, fontsize=8)
        S.epv_annotation(ax, tgt, cell="full_features", loc="upper right")
    axes[0].set_ylabel("observed / expected (O:E)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    print("saved", S.save(fig, HERE / "figures" / "fig_e1_calibration_drift"))


if __name__ == "__main__":
    main()
