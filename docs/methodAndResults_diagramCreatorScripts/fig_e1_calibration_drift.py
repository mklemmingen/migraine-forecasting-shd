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
            xticklab.append(f"{site}\ntest {obs:.0%} / train {trn:.0%}")
            for mi, m in enumerate(MODELS):
                r = d.get((tgt, site, m))
                if not r:
                    continue
                oe = float(r["oe_ratio"])
                ax.bar(si + (mi - 1.5) * w, oe, w, color=S.ARCH[m],
                       label=MLAB[m] if si == 0 else None, alpha=0.85)
                print(f"  {tgt:<9} {site:<10} {m:<16} O:E {oe:.2f}")
        ax.axhline(1.0, color="black", lw=1.2)
        ax.text(0.5, 1.02, "perfect (O:E = 1)", transform=ax.get_yaxis_transform(),
                ha="center", fontsize=7, va="bottom")
        ax.set_xticks(range(len(SITES))); ax.set_xticklabels(xticklab, fontsize=8)
        ax.set_title(f"{tgt}")
        ax.annotate("under-predicts", (1.0, 1.0), xytext=(-0.45, 1.32), fontsize=7,
                    color="#777", style="italic")
        ax.annotate("over-predicts", (1.0, 1.0), xytext=(-0.45, 0.55), fontsize=7,
                    color="#777", style="italic")
    axes[0].set_ylabel("observed / expected (O:E)")
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper center")
    fig.suptitle("Calibration drift on the held-out site tracks the base-rate gap",
                 y=1.02, fontsize=12)
    print("saved", S.save(fig, HERE / "figures" / "fig_e1_calibration_drift"))


if __name__ == "__main__":
    main()
