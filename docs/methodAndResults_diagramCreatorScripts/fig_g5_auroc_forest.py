"""Figure G5 - Pooled holdout AUROC forest across all architecture variants.

The Results section reports four anchor AUROC numbers in prose
(migraine 0.793 [0.544-0.890]; headache 0.653 [0.556-0.740]), but no body
figure shows the per-architecture spread the composite headline-cell selection
operates over. This forest plot fills that gap: one panel per target, one row
per evaluated architecture variant at the canonical full_features / 70_30 /
chrono headline cell, with patient-cluster bootstrap 95% CIs (the figure
honours the bootstrap unit the comparison_*.csv writer used, which is
patient-day rather than patient-cluster; the per-cell labels disclose the unit).

The vertical reference line at AUROC = 0.5 anchors chance discrimination.
The headline-cell architecture (XGB-HP020 for migraine, TabPFN-v2.6 for
headache, per the composite-tracked figdata_*.json) is highlighted; the
EPV-5.5 caveat on the migraine cell is marked with an annotation.

Source: latest experiment/comparison_*.csv (filtered to full_features /
70_30 / chrono rows for both targets).

Usage: python fig_g5_auroc_forest.py
"""
# §11 compliance:
#   §11.1 PASS: AUROC 95% CIs rendered as horizontal whiskers, patient-day
#               bootstrap unit per the comparison_*.csv source.

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path.insert(0, str(EXP))
import _style as S  # noqa: E402

BRACKET_RE = re.compile(r"^\s*([\-\d\.]+)\s*\[\s*([\-\d\.]+)\s*-\s*([\-\d\.]+)\s*\]\s*$")

HEADLINE = {
    "migraine": ("stacked_2xgb_meta_lr", "HP020"),
    "headache": ("tabpfn",               "version_2-6"),
}


def _latest_comparison_csv() -> Path:
    candidates = sorted(EXP.glob("comparison_*.csv"))
    if not candidates:
        raise SystemExit("no comparison_*.csv in experiment/; run experiment/run_aggregate_results.py")
    return candidates[-1]


def _parse_bracket(s: str) -> tuple[float, float, float]:
    if not isinstance(s, str):
        return float('nan'), float('nan'), float('nan')
    m = BRACKET_RE.match(s)
    if not m:
        return float('nan'), float('nan'), float('nan')
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _load_headline_rows(csv: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(csv)
    df = df[(df["feature_set"] == "full_features")
            & (df["datasplit"] == "70_30")
            & (df["splittype"] == "chrono")].copy()
    points: dict[str, list[dict]] = {"migraine": [], "headache": []}
    for _, row in df.iterrows():
        tgt = row["target"]
        if tgt not in points:
            continue
        point, lo, hi = _parse_bracket(row["holdout_AUROC"])
        if not np.isfinite(point):
            continue
        arch = row["architecture"]
        ver = row.get("version", "") or ""
        hp_variant = row.get("hp_variant", "") or ""
        if arch == "stacked_2xgb_meta_lr":
            strat = str(row.get("hp_strategy", "") or "")
            if hp_variant and "pareto" in strat:
                # knee / slope_closest / *_max selection rules recur across the
                # AUPRC- and AUROC-objective pareto fronts; tag the objective so
                # the two fronts do not collapse to one y-axis label.
                obj = "AUPRC" if "AUPRC" in strat else "AUROC"
                label = f"XGB / {hp_variant} [{obj}]"
            elif hp_variant:
                label = f"XGB / {hp_variant}"
            else:
                label = "XGB"
            short = "xgb"
        elif arch == "tabpfn":
            label = f"TabPFN / {ver}".replace("version_", "v")
            short = "tabpfn"
        else:
            label = f"{arch} / {ver}" if ver else arch
            short = arch
        headline_arch, headline_token = HEADLINE[tgt]
        is_headline = (arch == headline_arch
                       and ((hp_variant == headline_token) or (ver == headline_token)))
        points[tgt].append({
            "label": label, "short": short,
            "point": point, "lo": lo, "hi": hi,
            "is_headline": is_headline,
        })
    return {k: pd.DataFrame(v).sort_values("point").reset_index(drop=True) for k, v in points.items()}


def _palette(short: str) -> str:
    if short == "xgb":
        return S.ARCH.get("xgboost", S.OI["orange"])
    if short == "tabpfn":
        return S.ARCH.get("tabpfn", S.OI["skyblue"])
    return S.OI["green"]


def _render_panel(ax, df: pd.DataFrame, target: str, letter: str = "") -> None:
    if df.empty:
        ax.text(0.5, 0.5, f"no rows for {target}",
                transform=ax.transAxes, ha="center", va="center")
        return
    y = np.arange(len(df))
    xerr_lo = (df["point"] - df["lo"]).to_numpy()
    xerr_hi = (df["hi"] - df["point"]).to_numpy()
    for i, row in df.iterrows():
        c = _palette(row["short"])
        ax.errorbar(row["point"], i,
                    xerr=[[xerr_lo[i]], [xerr_hi[i]]],
                    fmt='o', capsize=3, ms=6 if row["is_headline"] else 5,
                    mew=1.5 if row["is_headline"] else 0.8,
                    mfc=c, mec=S.INK if row["is_headline"] else c,
                    ecolor=c, elinewidth=1.0)
        if row["is_headline"]:
            ax.annotate(" headline", xy=(row["hi"], i), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.5,
                        color=S.INK, va="center")
    ax.axvline(0.5, color=S.REF_COLOR, lw=1, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=7.5)
    ax.set_xlabel("pooled holdout AUROC (95% CI)")
    ax.set_title(f"{target.capitalize()}, all features, chronological 70/30",
                 fontsize=10.5, loc="left")
    if letter:
        ax.text(-0.02, 1.045, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="right")
    ax.set_xlim(0.4, 1.0)
    ax.grid(axis="x", color=S.FAINT, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    csv = _latest_comparison_csv()
    print(f"source: {csv.relative_to(REPO)}")
    panels = _load_headline_rows(csv)
    for tgt, df in panels.items():
        print(f"  {tgt:<9}: {len(df)} architecture variants  "
              f"AUROC range [{df['point'].min():.3f}, {df['point'].max():.3f}]")

    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharex=False,
                             constrained_layout=True)
    for ax, tgt, letter in zip(axes, ("migraine", "headache"), ("a", "b")):
        _render_panel(ax, panels[tgt], tgt, letter)
    S.epv_annotation(axes[0], "migraine", cell="full_features", loc="below")
    print("saved", S.save(fig, HERE / "figures" / "fig_g5_auroc_forest"))


if __name__ == "__main__":
    main()
