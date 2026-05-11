"""
dataset_analysis.py - builds dataset_analysis.pdf for one target mode.

Public API: run_dataset_analysis(diary_df, target_mode, ...)
"""
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from .style import apply_theme, PALETTE, POS_COLOR, NEG_COLOR

log = logging.getLogger(__name__)

# ── feature groups shown on the distribution pages ─────────────────────────
FEATURE_GROUPS = {
    "Migraine History": [
        "migraine_yesterday", "migraine_rate_last3", "migraine_rate_last7",
        "headache_free_streak", "days_since_last_migraine",
    ],
    "Sleep Features": [
        "lack_of_sleep_today", "oversleeping_today", "any_sleep_issue_today",
        "sleep_debt_3day", "sleep_disruption_today", "sleep_variability_7day",
        "recent_weekend_sleep_issues",
    ],
    "Triggers & Lifestyle": [
        "stress_today", "stress_drop_today", "consecutive_stress_days",
        "weather_change_today", "consecutive_weather_changes",
        "weather_instability_3day", "weather_change_yesterday",
        "weather_headache_interaction", "irregular_meals_today",
        "overeating_today", "excessive_caffeine_today", "alcohol_today",
        "travel_today", "consecutive_trigger_days",
        "exercise_today", "no_exercise_today",
        "consecutive_exercise_days", "consecutive_sedentary_days",
        "exercise_days_7day", "physical_fatigue_today",
        "emotional_changes_today", "noise_today",
        "specific_smells_today", "menstruation_today", "ovulation_today",
    ],
    "Low-Prevalence Restored Triggers": [
        "exercise_as_trigger_today", "sunlight_today",
        "inappropriate_lighting_today", "excessive_smoking_today",
        "cheese_chocolate_today",
    ],
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _reconstruct_today(df: pd.DataFrame, target_col: str, patient_col: str) -> pd.DataFrame:
    """Return df with migraine_today column (lag-1 of target per patient)."""
    df = df.copy()
    df["migraine_today"] = (
        df.groupby(patient_col)[target_col].shift(1)
    )
    df = df.dropna(subset=["migraine_today"])
    df["migraine_today"] = df["migraine_today"].astype(int)
    return df


def _ensure_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df


# ── page builders ────────────────────────────────────────────────────────────

def _page_cohort_summary(pdf: PdfPages, df: pd.DataFrame,
                         target_mode: str, target_col: str,
                         patient_col: str, date_col: str) -> None:
    log.info("Building page 1: cohort summary")
    apply_theme()

    n_rows = len(df)
    n_patients = df[patient_col].nunique()
    date_min = df[date_col].min().date()
    date_max = df[date_col].max().date()
    n_pos = df[target_col].sum()
    pos_rate = n_pos / n_rows
    n_features = df.shape[1]

    rows_per_patient = (
        df.groupby(patient_col).size().sort_values(ascending=False)
    )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 9),
        gridspec_kw={"height_ratios": [1, 2.5]},
    )
    fig.suptitle(f"Dataset Analysis - {target_mode}", fontsize=16, fontweight="bold")

    # summary table
    ax_tbl = axes[0]
    ax_tbl.axis("off")
    table_data = [
        ["Total rows", f"{n_rows:,}"],
        ["Total patients", str(n_patients)],
        ["Date range", f"{date_min} → {date_max}"],
        ["Positive rate", f"{pos_rate:.1%}  ({int(n_pos):,} events)"],
        ["Feature columns", str(n_features)],
    ]
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.35, 0.55],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0072B2")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")

    # bar chart rows per patient
    ax_bar = axes[1]
    x = np.arange(len(rows_per_patient))
    ax_bar.bar(x, rows_per_patient.values, color=PALETTE[0], edgecolor="none")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(rows_per_patient.index, rotation=90, fontsize=7)
    ax_bar.set_xlabel("Patient ID", fontsize=10)
    ax_bar.set_ylabel("Diary rows", fontsize=10)
    ax_bar.set_title("Diary rows per patient (sorted descending)", fontsize=11)
    sns.despine(ax=ax_bar)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close("all")


def _page_target_distribution(pdf: PdfPages, df: pd.DataFrame,
                              target_mode: str, target_col: str,
                              patient_col: str) -> None:
    log.info("Building page 2: target distribution")
    apply_theme()

    n_rows = len(df)
    n_pos = int(df[target_col].sum())
    n_neg = n_rows - n_pos
    overall_rate = n_pos / n_rows

    per_patient = df.groupby(patient_col)[target_col].agg(["sum", "count"])
    per_patient["rate"] = per_patient["sum"] / per_patient["count"]
    per_patient = per_patient.sort_values("rate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"Target Distribution - {target_mode}  (overall rate: {overall_rate:.1%})",
        fontsize=14, fontweight="bold",
    )

    # pie
    ax_pie = axes[0]
    ax_pie.pie(
        [n_neg, n_pos],
        labels=[f"Negative ({n_neg:,})", f"Positive ({n_pos:,})"],
        colors=[NEG_COLOR, POS_COLOR],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax_pie.set_title("Overall class split", fontsize=11)

    # per-patient rate histogram
    ax_hist = axes[1]
    rates = per_patient["rate"].values
    x = np.arange(len(rates))
    ax_hist.bar(x, rates, color=PALETTE[1], edgecolor="none")
    ax_hist.axhline(overall_rate, color=POS_COLOR, linewidth=1.5,
                    linestyle="--", label=f"Overall {overall_rate:.1%}")
    ax_hist.set_xlabel("Patient (sorted by rate)", fontsize=10)
    ax_hist.set_ylabel("Positive rate", fontsize=10)
    ax_hist.set_title(
        f"Per-patient positive rate\n"
        f"min={rates.min():.1%}  median={np.median(rates):.1%}  max={rates.max():.1%}",
        fontsize=11,
    )
    ax_hist.legend(fontsize=9)
    sns.despine(ax=ax_hist)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close("all")


def _page_gantt(pdf: PdfPages, df: pd.DataFrame, target_mode: str,
                target_col: str, patient_col: str, date_col: str) -> None:
    log.info("Building page 3: cohort timeline")
    apply_theme()

    summary = (
        df.groupby(patient_col)
        .agg(
            first_date=(date_col, "min"),
            last_date=(date_col, "max"),
            n_events=(target_col, "sum"),
            n_rows=(target_col, "count"),
        )
        .reset_index()
        .sort_values("first_date")
    )
    summary["density"] = summary["n_events"] / summary["n_rows"]

    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(summary["density"].min(), summary["density"].max())

    fig, ax = plt.subplots(figsize=(12, max(6, len(summary) * 0.28)))
    fig.suptitle(f"Cohort Timeline - {target_mode}", fontsize=14, fontweight="bold")

    for i, row in enumerate(summary.itertuples()):
        color = cmap(norm(row.density))
        ax.barh(
            i,
            (row.last_date - row.first_date).days,
            left=row.first_date,
            height=0.7,
            color=color,
            edgecolor="none",
        )

    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary[patient_col], fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title("Horizontal bar = diary coverage; colour = event density", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Event density", fontsize=9)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close("all")


def _pages_patient_timelines(pdf: PdfPages, df: pd.DataFrame,
                              target_mode: str, patient_col: str,
                              date_col: str) -> None:
    """Small-multiples per-patient event timelines, 6 per page."""
    log.info("Building patient timeline pages")
    apply_theme()

    PER_PAGE = 6
    event_counts = (
        df.groupby(patient_col)["migraine_today"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "events", "count": "rows"})
        .sort_values("events", ascending=False)
        .reset_index()
    )
    patients = event_counts[patient_col].tolist()

    for page_start in range(0, len(patients), PER_PAGE):
        chunk = patients[page_start: page_start + PER_PAGE]
        n = len(chunk)
        ncols = 2
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9))
        axes = np.array(axes).flatten()
        fig.suptitle(
            f"Per-patient Event Timelines - {target_mode}"
            f"  (patients {page_start + 1}–{page_start + n})",
            fontsize=12, fontweight="bold",
        )

        for idx, pid in enumerate(chunk):
            ax = axes[idx]
            pdata = (
                df[df[patient_col] == pid]
                .sort_values(date_col)
                [[date_col, "migraine_today"]]
                .copy()
            )
            info = event_counts[event_counts[patient_col] == pid].iloc[0]
            rate = info["events"] / info["rows"] if info["rows"] > 0 else 0

            # gray shading for gaps > 1 day
            pdata["gap"] = (
                pdata[date_col].diff().dt.days.fillna(1) > 1
            )
            for _, gr in pdata[pdata["gap"]].iterrows():
                ax.axvspan(
                    gr[date_col] - pd.Timedelta(days=1),
                    gr[date_col],
                    color="lightgray", alpha=0.5, zorder=0,
                )

            dates_neg = pdata[pdata["migraine_today"] == 0][date_col]
            dates_pos = pdata[pdata["migraine_today"] == 1][date_col]

            for d in dates_neg:
                ax.vlines(d, 0, 0.4, colors=NEG_COLOR, linewidth=0.6, alpha=0.5)
            for d in dates_pos:
                ax.vlines(d, 0, 1.0, colors=POS_COLOR, linewidth=1.2)

            ax.set_title(
                f"{pid}  |  {int(info['events'])} events  |  {rate:.1%}",
                fontsize=8,
            )
            ax.set_ylim(0, 1.2)
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=6)
            sns.despine(ax=ax, left=True)

        # hide unused axes
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close("all")


def _pages_feature_distributions(pdf: PdfPages, df: pd.DataFrame,
                                  target_mode: str, target_col: str) -> None:
    """One page per feature group showing histograms/bars split by class."""
    log.info("Building feature distribution pages")
    apply_theme()

    cols_in_df = set(df.columns)

    for group_name, features in FEATURE_GROUPS.items():
        present = [f for f in features if f in cols_in_df]
        if not present:
            continue

        n = len(present)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, nrows * 3)))
        axes = np.array(axes).flatten()
        fig.suptitle(
            f"Feature Distributions - {group_name}  ({target_mode})",
            fontsize=12, fontweight="bold",
        )

        df_pos = df[df[target_col] == 1]
        df_neg = df[df[target_col] == 0]

        for idx, feat in enumerate(present):
            ax = axes[idx]
            vals_pos = df_pos[feat].dropna()
            vals_neg = df_neg[feat].dropna()
            unique_vals = df[feat].dropna().nunique()

            if unique_vals <= 10:
                # binary / categorical: side-by-side bar
                categories = sorted(df[feat].dropna().unique())
                neg_counts = [len(vals_neg[vals_neg == c]) for c in categories]
                pos_counts = [len(vals_pos[vals_pos == c]) for c in categories]
                x = np.arange(len(categories))
                w = 0.35
                ax.bar(x - w / 2, neg_counts, w, label="Negative", color=NEG_COLOR)
                ax.bar(x + w / 2, pos_counts, w, label="Positive", color=POS_COLOR)
                ax.set_xticks(x)
                ax.set_xticklabels([str(c) for c in categories], fontsize=8)

                total_pos = len(vals_pos)
                if total_pos > 0:
                    prevalence = total_pos / (len(vals_neg) + total_pos)
                    ax.set_title(f"{feat}\nprevalence={prevalence:.1%}", fontsize=8)
                else:
                    ax.set_title(feat, fontsize=8)
            else:
                # continuous: overlapping histograms
                bins = min(30, unique_vals)
                ax.hist(vals_neg, bins=bins, alpha=0.6, color=NEG_COLOR,
                        label="Negative", density=True)
                ax.hist(vals_pos, bins=bins, alpha=0.6, color=POS_COLOR,
                        label="Positive", density=True)
                ax.set_title(feat, fontsize=8)

            ax.set_xlabel("")
            ax.set_ylabel("Count", fontsize=7)
            if idx == 0:
                ax.legend(fontsize=7)
            sns.despine(ax=ax)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close("all")


def _page_temporal_dependence(pdf: PdfPages, df: pd.DataFrame,
                               target_mode: str, patient_col: str) -> None:
    """ACF and conditional probability of migraine_today."""
    log.info("Building temporal dependence page")
    apply_theme()

    from statsmodels.tsa.stattools import acf

    MAX_LAGS = 14
    COND_LAGS = [1, 2, 3, 7]
    overall_rate = df["migraine_today"].mean()

    # per-patient ACF then average - every patient contributes
    patient_acfs = []
    for pid, grp in df.groupby(patient_col):
        series = grp.sort_values("date")["migraine_today"].values
        if np.var(series) == 0:
            # constant series: ACF is 0 at all lags by definition
            patient_acfs.append(np.zeros(MAX_LAGS))
            continue
        usable_lags = min(MAX_LAGS, len(series) - 2)
        a = acf(series, nlags=usable_lags, fft=True, alpha=None)
        lags_acf = a[1:]  # skip lag-0
        if usable_lags < MAX_LAGS:
            lags_acf = np.pad(lags_acf, (0, MAX_LAGS - usable_lags), constant_values=0.0)
        patient_acfs.append(lags_acf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Temporal Dependence - {target_mode}",
        fontsize=13, fontweight="bold",
    )

    ax_acf = axes[0]
    if patient_acfs:
        mean_acf = np.mean(patient_acfs, axis=0)
        std_acf = np.std(patient_acfs, axis=0)
        lags = np.arange(1, MAX_LAGS + 1)
        ci = 1.96 / np.sqrt(len(patient_acfs))

        ax_acf.bar(lags, mean_acf, color=PALETTE[0], alpha=0.8)
        ax_acf.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                             alpha=0.25, color=PALETTE[0])
        ax_acf.axhline(ci, color="gray", linestyle="--", linewidth=1)
        ax_acf.axhline(-ci, color="gray", linestyle="--", linewidth=1)
        ax_acf.axhline(0, color="black", linewidth=0.8)
        ax_acf.set_xlabel("Lag (days)", fontsize=10)
        ax_acf.set_ylabel("Mean ACF", fontsize=10)
        ax_acf.set_title("Population-averaged ACF of migraine_today", fontsize=11)
        ax_acf.set_xticks(lags)
        lag1_acf = mean_acf[0]
    else:
        ax_acf.text(0.5, 0.5, "Insufficient data for ACF",
                    ha="center", va="center", transform=ax_acf.transAxes)
        lag1_acf = float("nan")

    # conditional probability bars
    ax_cond = axes[1]
    cond_probs = []
    for k in COND_LAGS:
        shifted = df.groupby(patient_col)["migraine_today"].shift(k)
        mask = shifted == 1
        if mask.sum() > 0:
            p = df.loc[mask, "migraine_today"].mean()
        else:
            p = float("nan")
        cond_probs.append(p)

    bars = ax_cond.bar(
        [f"k={k}" for k in COND_LAGS],
        cond_probs,
        color=PALETTE[1],
        edgecolor="none",
    )
    ax_cond.axhline(overall_rate, color=POS_COLOR, linestyle="--", linewidth=1.5,
                    label=f"Unconditional rate ({overall_rate:.1%})")
    for bar, p in zip(bars, cond_probs):
        if not math.isnan(p):
            ax_cond.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{p:.1%}", ha="center", va="bottom", fontsize=9,
            )
    ax_cond.set_ylabel("P(event today | event k days ago)", fontsize=10)
    ax_cond.set_title("Conditional probability by lag", fontsize=11)
    ax_cond.legend(fontsize=9)
    sns.despine(ax=ax_cond)

    # verdict
    if not math.isnan(lag1_acf):
        strength = "strong" if abs(lag1_acf) > 0.2 else ("weak" if abs(lag1_acf) > 0.05 else "no")
        dependence = "dependence" if abs(lag1_acf) > 0.05 else "independence"
        verdict = (
            f"migraine_today shows {strength} lag-1 autocorrelation "
            f"(ACF={lag1_acf:.2f}), suggesting {dependence}."
        )
        fig.text(0.5, 0.01, verdict, ha="center", fontsize=10, style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    pdf.savefig(fig, dpi=150)
    plt.close("all")


# ── public entry point ───────────────────────────────────────────────────────

def run_dataset_analysis(
    diary_df: pd.DataFrame,
    target_mode: str,
    target_col: str = "migraine_target",
    patient_col: str = "patient_id",
    date_col: str = "date",
    output_path: Path = None,
) -> None:
    """Build dataset_analysis.pdf for one target mode."""
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent
            / "processed" / target_mode / "dataset_analysis.pdf"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("run_dataset_analysis: target_mode=%s  output=%s", target_mode, output_path)

    df = _ensure_date(diary_df.copy(), date_col)
    df_today = _reconstruct_today(df, target_col, patient_col)

    with PdfPages(output_path) as pdf:
        _page_cohort_summary(pdf, df, target_mode, target_col, patient_col, date_col)
        _page_target_distribution(pdf, df, target_mode, target_col, patient_col)
        _page_gantt(pdf, df, target_mode, target_col, patient_col, date_col)
        _pages_patient_timelines(pdf, df_today, target_mode, patient_col, date_col)
        _pages_feature_distributions(pdf, df, target_mode, target_col)
        _page_temporal_dependence(pdf, df_today, target_mode, patient_col)

    log.info("Saved %s", output_path)
    plt.close("all")
