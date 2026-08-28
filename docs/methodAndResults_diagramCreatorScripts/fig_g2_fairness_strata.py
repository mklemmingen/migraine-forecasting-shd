"""Figure 6 (working-notes ID: G2) - fairness stratum disaggregation of the
within-person C-statistic.

Body-facing forest plot of the Results section's stratum disaggregation that
the manuscript currently cites in text + CSV form only (the canonical CSV at
experiment/_eval/_special/within_person_stratum_*.csv). Two panels: panel a
headache target, panel b migraine target. Per panel, each stratum row carries
the three within-architecture point estimates (XGBoost stack, TabPFN,
window-MLP) with horizontal 95% CI whiskers; strata with n_estimable below
the five-patient reporting floor (migraine male n = 1 and migraine
below-7.2%-base-rate n = 3, across all three architectures) are surfaced
quantitatively with the n count and a "below floor" annotation rather than
collapsed into the pooled view. The vertical chance reference at 0.5 anchors
the near-chance reading visually.

Data source: experiment/_eval/_special/within_person_stratum_*.csv (latest
timestamp). The 42-row CSV covers (2 targets) x (3 architectures) x (1
pooled + 2 site + 2 sex + 2 base_rate strata) = 42 rows; below_floor flag is
True for the migraine male (n = 1) and migraine below-7.2%-base-rate (n = 3)
rows where within_c, ci_low, ci_high are all NaN by construction.

Usage: python fig_g2_fairness_strata.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiment"))
import _style as S  # noqa: E402

CSV_GLOB = REPO / "experiment" / "_eval" / "_special" / "within_person_stratum_*.csv"

# Architecture order + colour: matches body figures fig_e1 LOSO and fig_g1
# SHAP panels for cross-figure architecture-identity consistency.
ARCH_ORDER = ["XGBoost stack", "TabPFN", "window-MLP"]
ARCH_MARKER = {"XGBoost stack": "o", "TabPFN": "s", "window-MLP": "^"}

# Stratum row order per panel (top-down): pooled, site, sex, base_rate.
# Each entry: (stratum_label_for_y_axis, csv_stratum_dim, csv_stratum_value)
def _strata_rows(target: str) -> list[tuple[str, str, str]]:
    br_threshold = "25.0%" if target == "headache" else "7.2%"
    return [
        ("pooled (all estimable)",      "(pooled)",  "(pooled)"),
        ("site = Uijeongbu",            "site",      "site=Uijeongbu"),
        ("site = Dongtan",              "site",      "site=Dongtan"),
        ("sex = female",                "sex",       "sex=female"),
        ("sex = male",                  "sex",       "sex=male"),
        (f"base rate above {br_threshold}", "base_rate", f"base_rate above {br_threshold}"),
        (f"base rate below {br_threshold}", "base_rate", f"base_rate below {br_threshold}"),
    ]


def _resolve_csv() -> Path:
    matches = sorted(CSV_GLOB.parent.glob(CSV_GLOB.name))
    if not matches:
        raise SystemExit(
            f"no stratum CSV at {CSV_GLOB}; run the §3.6 stratum disaggregation "
            f"pass to generate one"
        )
    return matches[-1]


def _draw_panel(ax, df: pd.DataFrame, target: str) -> None:
    """Render the per-stratum × per-architecture forest plot for one target."""
    rows = _strata_rows(target)
    sub = df[df["target"] == target].copy()

    # Y-axis: 7 strata x 3 architectures = 21 horizontal slots, top-down.
    # Within each stratum, the 3 architecture rows are vertically jittered so
    # the markers and CIs do not overprint each other at the stratum y-coord.
    n_strata = len(rows)
    arch_jitter = {"XGBoost stack": +0.22, "TabPFN": 0.0, "window-MLP": -0.22}

    for stratum_idx, (y_label, dim, val) in enumerate(rows):
        # Light grey horizontal divider between adjacent stratum groups
        if stratum_idx > 0:
            ax.axhline(stratum_idx - 0.5, color=S.FAINT, lw=0.5, zorder=0)

        # A below-floor stratum is below_floor across all three architectures
        # (within_c/ci_low/ci_high are NaN by construction), so a single
        # centred annotation at the stratum y-coord carries the n count
        # without the per-architecture jitter that would otherwise push the
        # bottom copy onto the x-axis. A plotted symbol would mis-read as an
        # estimate, but C is not estimable when n_estimable < 5.
        stratum_rows = sub[(sub["stratum_dim"] == dim) & (sub["stratum"] == val)]
        if len(stratum_rows) and bool(stratum_rows.iloc[0]["below_floor"]):
            n_est = int(stratum_rows.iloc[0]["n_estimable"])
            # Starts clear of the dashed chance rule at 0.50, which used to run
            # straight through the first characters, and in INK rather than the
            # muted grey: this note explains a MISSING estimate, so it is the
            # most important thing in its row, not the least.
            ax.text(0.535, stratum_idx,
                    f"n = {n_est}, not estimable (below 5-patient floor)",
                    va="center", ha="left", fontsize=7.4, color=S.INK)
            continue

        for arch in ARCH_ORDER:
            row = sub[(sub["architecture"] == arch) &
                      (sub["stratum_dim"] == dim) &
                      (sub["stratum"] == val)]
            if len(row) != 1:
                continue
            r = row.iloc[0]
            y = stratum_idx + arch_jitter[arch]
            colour = S.arch_color(arch)
            n_est = int(r["n_estimable"])
            wc = float(r["within_c"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            ax.errorbar([wc], [y], xerr=[[wc - lo], [hi - wc]], fmt=ARCH_MARKER[arch],
                        color=colour, ecolor=colour, elinewidth=0.9, capsize=2.5,
                        markersize=5, mew=0.6, mfc=colour, mec=S.INK, zorder=4)
            ax.text(hi + 0.012, y, f"{wc:.3f} (n={n_est})",
                    va="center", fontsize=7, color=S.INK)

    # Chance reference at 0.5
    ax.axvline(0.5, color=S.REF_COLOR, lw=S.REF_LW, ls="--", alpha=0.6, zorder=1)

    ax.set_yticks(range(n_strata))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    # Explicit inverted limits with 0.7 padding past the first/last stratum so
    # the jittered per-architecture markers (and any text) clear the panel
    # edges and the x-axis rather than relying on matplotlib's data-only
    # autoscale, which ignores text height.
    ax.set_ylim(n_strata - 1 + 0.7, -0.7)
    ax.set_xlim(0.35, 0.95)
    ax.set_xlabel("within-person C-statistic (95% CI)", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    S.apply()
    csv_path = _resolve_csv()
    df = pd.read_csv(csv_path)
    print(f"  source {csv_path.name} ({len(df)} stratum rows)")

    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 5.2), sharey=False)
    # Wide inter-panel gutter: panel b carries its own y-tick labels (the
    # base-rate threshold and the sex=male estimability differ between
    # targets), and those labels grow leftward from panel b's left spine.
    # A roomy wspace keeps them off panel a's right-hand value annotations.
    fig.subplots_adjust(wspace=0.6)
    for ax, ltr in zip(axes.ravel(), "ab"):
        S.panel_label(ax, ltr)
    _draw_panel(axes[0], df, "headache")
    _draw_panel(axes[1], df, "migraine")

    # Shared legend for the three architectures (the only legend the figure needs)
    handles = []
    for arch in ARCH_ORDER:
        handles.append(plt.Line2D([], [], marker=ARCH_MARKER[arch],
                                  color=S.arch_color(arch), lw=0,
                                  markersize=6, mew=0.6, mec=S.INK,
                                  label=arch))
    handles.append(plt.Line2D([], [], color=S.REF_COLOR, lw=S.REF_LW, ls="--",
                              alpha=0.6, label="chance 0.5"))
    fig.legend(handles=handles, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), frameon=False)

    out = HERE / "figures" / "fig_g2_fairness_strata"
    print("saved", S.save(fig, out))


if __name__ == "__main__":
    main()
