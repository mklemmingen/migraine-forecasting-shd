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
MCOL = {"pooled_lr": "#666666", "add0_stacked": "#1b9e77",
        "add1_tabpfn": "#7570b3", "add4_window_mlp": "#d95f02"}
SITES = ["uijeongbu", "dongtan"]
# internal CV-OOF within-person (TabPFN, Figure C2 / Addition 5 Section 9b)
INTERNAL = {"headache": 0.538, "migraine": 0.573}


def _latest():
    return max((EXP / "5").glob("external_site_summary_*.csv"), key=lambda p: p.stat().st_mtime)


def main():
    S.apply()
    rows = list(csv.DictReader(open(_latest())))
    d = {(r["target"], r["held_out"], r["model"]): r for r in rows}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=True)
    for ax, tgt in zip(axes, ("headache", "migraine")):
        for si, site in enumerate(SITES):
            for m in MODELS:
                r = d.get((tgt, site, m))
                if not r or r["within_cstat"] in ("", "nan"):
                    continue
                v = float(r["within_cstat"])
                ax.scatter(si + (rng.random() - 0.5) * 0.4, v, s=55, color=MCOL[m],
                           alpha=0.85, edgecolor="white", lw=0.5,
                           label=MLAB[m] if si == 0 else None)
                print(f"  {tgt:<9} {site:<10} {m:<16} within {v:.3f}")
        ax.axhline(0.5, color="black", lw=1.0, ls="--", label="chance")
        ax.axhline(INTERNAL[tgt], color=S.TARGET[tgt], lw=1.4,
                   label=f"internal within-person {INTERNAL[tgt]:.2f}")
        ax.set_xticks(range(len(SITES)))
        ax.set_xticklabels([f"held-out\n{s}" for s in SITES])
        ax.set_ylim(0.40, 0.75)
        ax.set_title(tgt)
    axes[0].set_ylabel("within-person C-statistic")
    axes[0].legend(fontsize=7, loc="upper center", ncol=2)
    fig.suptitle("Within-person discrimination stays near chance off-site", y=1.02, fontsize=12)
    print("saved", S.save(fig, HERE / "figures" / "fig_e2_within_person_sites"))


if __name__ == "__main__":
    main()
