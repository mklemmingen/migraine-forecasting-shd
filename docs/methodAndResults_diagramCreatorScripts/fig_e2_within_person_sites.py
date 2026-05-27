"""Figure E2 - within-person discrimination replicates off-site.

Precision-weighted within-person C-statistic on each held-out recruitment site,
per model and target, against chance (0.5) and the internal CV-out-of-fold
within-person estimate (Figure C2). The off-site values cluster around chance just
as internally - the near-chance per-patient result is reproduced on an independent
clinic, not a single-cohort artefact (docs/external_validation_site.md Section 7).
Reads the latest experiment/5/external_site_summary_*.csv.

Usage: python fig_e2_within_person_sites.py
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
                ax.scatter(si + (rng.random() - 0.5) * 0.55, v, s=55, color=S.ARCH[m],
                           alpha=0.85, edgecolor="white", lw=0.5,
                           label=MLAB[m] if si == 0 else None)
                print(f"  {tgt:<9} {site:<10} {m:<16} within {v:.3f}")
        S.refline(ax, y=0.5, label="chance")
        # neutral reference line + per-panel value, so the shared legend swatch
        # cannot mismatch the drawn colour (the internal estimate is target-specific)
        ax.axhline(INTERNAL[tgt], color=S.SOFT, lw=1.4)
        ax.text(0.02, INTERNAL[tgt], f"internal C {INTERNAL[tgt]:.2f}",
                transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color=S.SOFT)
        ax.set_xticks(range(len(SITES)))
        ax.set_xticklabels([f"held-out\n{s}" for s in SITES])
        ax.set_ylim(0.40, 0.85)
        ax.set_title(tgt)
    axes[0].set_ylabel("within-person C-statistic")
    axes[0].legend(fontsize=8, loc="upper center", ncol=2)
    fig.suptitle("Within-person discrimination stays near chance off-site", y=1.02)
    print("saved", S.save(fig, HERE / "figures" / "fig_e2_within_person_sites"))


if __name__ == "__main__":
    main()
