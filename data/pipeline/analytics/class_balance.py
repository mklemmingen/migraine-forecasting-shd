"""
class_balance.py - builds class_balance.pdf for one target mode.

Public API: run_class_balance(processed_dir, target_mode, ...)
"""
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from .style import apply_theme, PALETTE, POS_COLOR, NEG_COLOR

log = logging.getLogger(__name__)

RATIOS = ["70_15_15", "80_20", "70_30"]
STRATEGIES = ["chrono", "patient", "stratified"]
FOLDS = ["train", "val", "test"]


def _load_packages(processed_dir: Path, target_col: str) -> List[Dict]:
    """
    Returns a list of package dicts, each with:
      ratio, strategy, splits (dict fold→DataFrame), overall_rate
    Skips packages where no diary parquets are found.
    """
    processed_dir = Path(processed_dir)
    diary_full = processed_dir / "diary.parquet"
    overall_rate = None
    if diary_full.exists():
        full_df = pd.read_parquet(diary_full, columns=[target_col])
        overall_rate = full_df[target_col].mean()

    packages = []
    for ratio in RATIOS:
        for strategy in STRATEGIES:
            pkg_dir = processed_dir / ratio / strategy
            splits = {}
            for fold in FOLDS:
                fp = pkg_dir / f"diary_{fold}.parquet"
                if fp.exists():
                    splits[fold] = pd.read_parquet(fp, columns=[target_col])
            if splits:
                packages.append(
                    dict(
                        ratio=ratio,
                        strategy=strategy,
                        label=f"{ratio}/{strategy}",
                        splits=splits,
                        overall_rate=overall_rate,
                    )
                )
    return packages


def _split_stats(splits: Dict[str, pd.DataFrame], target_col: str) -> pd.DataFrame:
    """Return a DataFrame with rows=[fold], cols=[n_rows, n_events, rate]."""
    rows = []
    for fold in FOLDS:
        if fold not in splits:
            continue
        df = splits[fold]
        n = len(df)
        events = int(df[target_col].sum())
        rate = events / n if n > 0 else float("nan")
        rows.append({"fold": fold, "n_rows": n, "n_events": events, "rate": rate})
    return pd.DataFrame(rows).set_index("fold")


# ── page builders ────────────────────────────────────────────────────────────

def _page_overview_matrix(
    pdf: PdfPages,
    packages: List[Dict],
    target_mode: str,
    target_col: str,
    overall_rate: Optional[float],
) -> None:
    log.info("Building overview matrix page")
    apply_theme()

    rows = []
    for pkg in packages:
        stats = _split_stats(pkg["splits"], target_col)
        row = {"Package": pkg["label"]}
        for fold in FOLDS:
            if fold in stats.index:
                row[f"{fold}_rate"] = stats.loc[fold, "rate"]
                row[f"{fold}_events"] = stats.loc[fold, "n_events"]
            else:
                row[f"{fold}_rate"] = float("nan")
                row[f"{fold}_events"] = float("nan")
        rows.append(row)

    matrix_df = pd.DataFrame(rows).set_index("Package")

    rate_cols = [c for c in matrix_df.columns if c.endswith("_rate")]

    fig, ax = plt.subplots(figsize=(10, max(5, len(packages) * 0.7 + 2)))
    fig.suptitle(
        f"Class Balance Overview - {target_mode}\n"
        f"(green: within 5pp of overall; yellow: 5–10pp; red: >10pp)",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")

    event_cols = [c for c in matrix_df.columns if c.endswith("_events")]

    # package name is the first data column - no rowLabels so nothing overflows left
    col_labels = (
        ["Package (ratio / strategy)"]
        + [c.replace("_rate", " Rate").replace("_", " ").title() for c in rate_cols]
        + [c.replace("_events", " Events").replace("_", " ").title() for c in event_cols]
    )

    display_rows = []
    cell_colors = []
    for pkg_name, row in matrix_df.iterrows():
        dr = [pkg_name]
        dc = ["#e8eaf6"]   # soft indigo for the name column
        for col in rate_cols:
            rate = row[col]
            if math.isnan(rate):
                dr.append("-")
                dc.append("#f5f5f5")
            else:
                dr.append(f"{rate:.1%}")
                if overall_rate is not None:
                    delta = abs(rate - overall_rate)
                    if delta <= 0.05:
                        dc.append("#c8e6c9")
                    elif delta <= 0.10:
                        dc.append("#fff9c4")
                    else:
                        dc.append("#ffcdd2")
                else:
                    dc.append("white")
        for col in event_cols:
            v = row[col]
            dr.append("-" if math.isnan(v) else str(int(v)))
            dc.append("white")
        display_rows.append(dr)
        cell_colors.append(dc)

    tbl = ax.table(
        cellText=display_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)

    if overall_rate is not None:
        fig.text(
            0.5, 0.02,
            f"Overall diary rate: {overall_rate:.2%}",
            ha="center", fontsize=10, style="italic",
        )

    pdf.savefig(fig, dpi=150)
    plt.close("all")


def _page_package(pdf: PdfPages, pkg: Dict, target_mode: str,
                  target_col: str, overall_rate: Optional[float]) -> None:
    log.info("Building package page: %s", pkg["label"])
    apply_theme()

    splits = pkg["splits"]
    stats = _split_stats(splits, target_col)
    present_folds = list(stats.index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{pkg['label']} - {target_mode}",
        fontsize=13, fontweight="bold",
    )

    # left: stacked bar counts
    ax_bar = axes[0]
    x = np.arange(len(present_folds))
    neg_counts = [
        stats.loc[f, "n_rows"] - stats.loc[f, "n_events"] for f in present_folds
    ]
    pos_counts = [stats.loc[f, "n_events"] for f in present_folds]

    ax_bar.bar(x, neg_counts, color=NEG_COLOR, label="Negative", rasterized=True)
    ax_bar.bar(x, pos_counts, bottom=neg_counts, color=POS_COLOR, label="Positive", rasterized=True)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f.capitalize() for f in present_folds], fontsize=10
    )
    ax_bar.set_xlabel("Split", fontsize=10)
    ax_bar.set_ylabel("Number of diary rows", fontsize=10)
    ax_bar.set_title(
        "Absolute class counts per split\n(stacked: negative + positive)", fontsize=11
    )
    ax_bar.legend(fontsize=9, title="Class")
    for xi, fold in enumerate(present_folds):
        n = stats.loc[fold, "n_rows"]
        e = stats.loc[fold, "n_events"]
        ax_bar.text(xi, n + n * 0.01, f"n={n:,}\nevents={e}", ha="center",
                    va="bottom", fontsize=8)
    sns.despine(ax=ax_bar)

    # right: horizontal rate bars
    ax_rate = axes[1]
    rates = [stats.loc[f, "rate"] for f in present_folds]
    colors = [POS_COLOR if f == "test" else PALETTE[4] for f in present_folds]
    bars = ax_rate.barh(present_folds, rates, color=colors, edgecolor="none", rasterized=True)
    if overall_rate is not None:
        ax_rate.axvline(overall_rate, color="gray", linestyle="--", linewidth=1.5,
                        label=f"Overall {overall_rate:.1%}")
    for bar, rate in zip(bars, rates):
        ax_rate.text(
            rate + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{rate:.1%}", va="center", fontsize=9,
        )
    ax_rate.set_xlabel("Positive rate  (fraction of rows that are migraine events)", fontsize=10)
    ax_rate.set_ylabel("Split", fontsize=10)
    ax_rate.set_title(
        "Positive rate per split\n(dashed = overall diary rate)", fontsize=11
    )
    ax_rate.legend(fontsize=9)
    sns.despine(ax=ax_rate)

    # imbalance annotation
    if overall_rate is not None:
        notes = []
        for fold in present_folds:
            r = stats.loc[fold, "rate"]
            delta = r - overall_rate
            ratio_val = (1 - r) / r if r > 0 else float("nan")
            notes.append(f"{fold}: imbalance ratio {ratio_val:.1f}:1  (Δ={delta:+.1%})")
        fig.text(0.5, 0.01, "   |   ".join(notes), ha="center", fontsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    pdf.savefig(fig, dpi=200)
    plt.close("all")


def _page_cross_package(pdf: PdfPages, packages: List[Dict], target_mode: str,
                        target_col: str, overall_rate: Optional[float]) -> None:
    log.info("Building cross-package comparison page")
    apply_theme()

    labels = [pkg["label"] for pkg in packages]
    test_rates = []
    train_rates = []
    for pkg in packages:
        stats = _split_stats(pkg["splits"], target_col)
        test_rates.append(stats.loc["test", "rate"] if "test" in stats.index else float("nan"))
        train_rates.append(stats.loc["train", "rate"] if "train" in stats.index else float("nan"))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Cross-package Test Positive Rate - {target_mode}",
        fontsize=13, fontweight="bold",
    )

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, train_rates, w, color=PALETTE[4], label="Train rate", rasterized=True)
    ax.bar(x + w / 2, test_rates, w, color=POS_COLOR, label="Test rate", rasterized=True)

    if overall_rate is not None:
        ax.axhline(overall_rate, color="black", linestyle="--", linewidth=1.5,
                   label=f"Overall {overall_rate:.1%}")

    # highlight large train-test deltas
    for xi, (tr, te) in enumerate(zip(train_rates, test_rates)):
        if not (math.isnan(tr) or math.isnan(te)) and abs(tr - te) > 0.05:
            ax.annotate(
                f"Δ={abs(tr-te):.1%}",
                xy=(xi, max(tr, te)),
                xytext=(xi, max(tr, te) + 0.01),
                ha="center", fontsize=8, color="red",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Package  (split ratio / patient-assignment strategy)", fontsize=10)
    ax.set_ylabel("Positive rate  (fraction of rows that are migraine events)", fontsize=10)
    ax.set_title(
        "Train vs test positive rate across all 9 packages\n"
        "Red annotation = train-test delta > 5 percentage points",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close("all")


# ── public entry point ───────────────────────────────────────────────────────

def run_class_balance(
    processed_dir: Path,
    target_mode: str,
    target_col: str = "migraine_target",
    output_path: Path = None,
) -> None:
    """Build class_balance.pdf for one target mode."""
    processed_dir = Path(processed_dir)
    if output_path is None:
        output_path = processed_dir / "class_balance.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("run_class_balance: target_mode=%s  output=%s", target_mode, output_path)

    packages = _load_packages(processed_dir, target_col)
    if not packages:
        log.warning("No split parquets found under %s - skipping class_balance", processed_dir)
        return

    overall_rate = packages[0]["overall_rate"]

    with PdfPages(output_path) as pdf:
        _page_overview_matrix(pdf, packages, target_mode, target_col, overall_rate)
        for pkg in packages:
            _page_package(pdf, pkg, target_mode, target_col, overall_rate)
        _page_cross_package(pdf, packages, target_mode, target_col, overall_rate)

    log.info("Saved %s", output_path)
    plt.close("all")
