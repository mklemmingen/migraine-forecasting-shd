"""Figure G9 - within-family AUROC of all TabPFN variants on the canonical
cell.

Side-by-side hold-out AUROC bars for every released TabPFN variant on the
full_features / chronological / 70-30 cell, per target. The composite-sorted
selection rule treats AUROC differences below 0.02 as ties; this figure
visualises the within-family tie that the rule then resolves on calibration
distance rather than discrimination. The shaded band marks the +/-0.02
composite-tie zone anchored at the highest-AUROC variant per target. Per-bar
markers (circle / square / triangle / diamond / down-triangle / plus) and a
within-family brightness ramp give CVD/grayscale survival even though all bars
share the TabPFN reddish-purple family anchor.

This is the visual companion to the textual within-family-tie claim in
results_findings.md Section 3a Finding (2): the variant rank ordering within
the headache full_features/chronological tier carries no statistical signal.

Reads experiment/comparison_*.csv (no compute).

Usage: python fig_g9_tabpfn_family.py
"""
# §11 compliance: within-family TabPFN AUROC at the canonical migraine cell.
#   §11.1 PASS (95% CI xerr + 0.02-AUROC tie band rendered)
#   §11.3 caption: cohort+n + within-family-tie note in rendered title
#   §11.7 CRITICAL: this IS the migraine `full_features` cell; renderer
#       must overlay the EPV-5.5 marker per §11 -- helper enhancement pending
import re
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiment"

# Canonical ordering by release lineage (older first), so the within-family
# brightness ramp reads as "lighter = older variant, darker = newer variant".
VARIANTS = [
    ("version_2-5-auto", "TabPFN-v2.5a"),
    ("version_2-5-real", "TabPFN-v2.5r"),
    ("version_2-5-finetuned", "TabPFN-v2.5f"),
    ("version_2-6", "TabPFN-v2.6"),
    ("version_3-default", "TabPFN-v3d"),
    ("version_3-binary", "TabPFN-v3b"),
]
MARKERS = ["o", "s", "^", "D", "v", "P"]

_RE = re.compile(r"([\d.]+)\s*\[([\d.]+)\s*-\s*([\d.]+)\]")


def _parse_auroc(cell):
    """Parse '0.745 [0.656 - 0.825]' into (mean, lo, hi); None on failure."""
    if not isinstance(cell, str):
        return None
    m = _RE.match(cell.strip())
    if m is None:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _latest_csv():
    csvs = sorted(EXP.glob("comparison_*.csv"))
    if not csvs:
        raise SystemExit("no comparison_*.csv in experiment/")
    return csvs[-1]


def _shade(idx, total, anchor_rgb):
    """Within-family brightness ramp: oldest is lightest, newest is darkest."""
    light = np.array([0.94, 0.88, 0.94])
    anchor = np.array(anchor_rgb)
    t = 0.30 + 0.70 * idx / max(1, total - 1)
    return tuple(light + t * (anchor - light))


def main():
    S.apply()
    csv = _latest_csv()
    df = pd.read_csv(csv)
    print(f"  source {csv.name}")

    anchor_rgb = mc.to_rgb(S.ARCH.get("tabpfn", S.OI["purple"]))
    palette = [_shade(i, len(VARIANTS), anchor_rgb) for i in range(len(VARIANTS))]

    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.5))
    for _ax, _lt in zip(axes.ravel(), "abcd"):
        S.panel_label(_ax, _lt, x=-0.10, y=1.06)

    for ax, tgt in zip(axes, ("headache", "migraine")):
        q = df[(df["target"] == tgt) & (df["feature_set"] == "full_features")
               & (df["splittype"] == "chrono") & (df["datasplit"] == "70_30")
               & (df["architecture"] == "tabpfn")]
        rows = []
        for i, (csv_v, slug) in enumerate(VARIANTS):
            sub = q[q["version"] == csv_v]
            if sub.empty:
                continue
            parsed = _parse_auroc(sub["holdout_AUROC"].iloc[0])
            if parsed is None:
                continue
            mean, lo, hi = parsed
            rows.append((i, slug, mean, lo, hi))

        if not rows:
            ax.text(0.5, 0.5, f"no TabPFN data for {tgt}", ha="center",
                    transform=ax.transAxes)
            ax.set_title(tgt)
            continue

        # Sort by AUROC ascending so the highest is at the top of the panel
        rows.sort(key=lambda r: r[2])
        ys = np.arange(len(rows))
        means = [r[2] for r in rows]
        los = [r[2] - r[3] for r in rows]
        his = [r[4] - r[2] for r in rows]
        colors = [palette[r[0]] for r in rows]
        mks = [MARKERS[r[0]] for r in rows]
        slugs = [r[1] for r in rows]

        ax.barh(ys, means, height=0.62, color=colors,
                xerr=[los, his],
                error_kw={"elinewidth": 0.8, "capsize": 2})
        for y, m, mk, color in zip(ys, means, mks, colors):
            ax.scatter([m], [y], marker=mk, s=44, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)

        # +/- 0.02 composite-tie band anchored at the top-AUROC variant
        top = max(means)
        ax.axvspan(top - 0.02, top, color=S.MUTED, alpha=0.15, zorder=0)

        ax.set_yticks(ys)
        ax.set_yticklabels(slugs, fontsize=8)
        ax.set_xlim(0.5, max(0.85, max(r[4] for r in rows) + 0.03))
        ax.set_xlabel("hold-out AUROC (95% CI)")
        ax.set_title(tgt)
        S.refline(ax, x=0.5)
        S.epv_annotation(ax, tgt, cell="full_features", loc="lower right")
        print(f"  {tgt}: {len(rows)} variants, top {top:.3f}, span "
              f"{max(means) - min(means):+.3f}")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=S.MUTED, alpha=0.15,
                      label="0.02 composite-tie band"),
        plt.Line2D([], [], linestyle="--", color=S.REF_COLOR, lw=1,
                   label="chance (0.5)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle("Within-family AUROC across all TabPFN variants "
                 "(Park 2016 SHD, n=62; full / chrono / 70-30)", y=1.02)
    print(f"  saved {S.save(fig, HERE / 'figures' / 'fig_g9_tabpfn_family')}")


if __name__ == "__main__":
    main()
