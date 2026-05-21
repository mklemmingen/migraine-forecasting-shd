"""Gap-aware calendar-regular daily attack series per patient.

The SHD diary is recorded irregularly: many patient-day transitions span a
gap longer than one day. The analytics page computes its ACF on the
compacted record index, so a lag of k there means k records, not k
calendar days. This module reindexes each patient to a calendar-regular
daily index with explicit missing days, so every downstream lag is a true
calendar lag.

Missing days are NaN, not zero: an absent diary entry is unknown, not a
known no-attack day, and coding it as zero would fabricate temporal
structure. Downstream estimators decide how to treat the NaNs (pairwise
deletion for the ACF, gap times for the recurrent-event models).

The attack indicator is the per-day binary outcome column (``*_target``)
from the unsplit ``data/processed/<target>/diary.parquet``.
Autocorrelation, transition and burstiness summaries are invariant to a
uniform one-day labelling shift, so the today-vs-next-day labelling of the
target does not affect them; the self-excitation regression, which aligns
the attack with same-day triggers, handles the alignment explicitly.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PATIENT_COL = "patient_id"
DATE_COL = "date"


def diary_path(target: str) -> Path:
    """Path to the unsplit diary parquet for a target (migraine|headache)."""
    return REPO / "data" / "processed" / target / "diary.parquet"


def attack_column(df: pd.DataFrame) -> str:
    """Return the single binary attack-outcome column (ends in ``_target``)."""
    cands = [c for c in df.columns if c.endswith("_target")]
    if len(cands) != 1:
        raise ValueError(f"expected exactly one *_target column, found {cands}")
    return cands[0]


def load_diary(target: str) -> pd.DataFrame:
    """Load the unsplit diary parquet for a target, sorted by patient/date."""
    df = pd.read_parquet(diary_path(target))
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values([PATIENT_COL, DATE_COL]).reset_index(drop=True)


def build_patient_series(df: pd.DataFrame, attack_col: str):
    """Per-patient calendar-regular daily attack series.

    Returns ``{patient_id: pd.Series}`` where each series has a daily
    ``DatetimeIndex`` from the patient's first to last recorded day, values
    in ``{0.0, 1.0}`` on recorded days and ``NaN`` on missing calendar days.
    """
    out = {}
    for pid, grp in df.groupby(PATIENT_COL, sort=True):
        g = (grp[[DATE_COL, attack_col]]
             .dropna(subset=[DATE_COL])
             .drop_duplicates(subset=[DATE_COL], keep="last")
             .sort_values(DATE_COL))
        if g.empty:
            continue
        s = pd.Series(g[attack_col].to_numpy(dtype=float),
                      index=pd.DatetimeIndex(g[DATE_COL]))
        full = pd.date_range(s.index.min(), s.index.max(), freq="D")
        out[pid] = s.reindex(full)
    return out


def gap_summary(df: pd.DataFrame) -> dict:
    """Cohort gap statistics from the observed (pre-reindex) record dates:
    transitions longer than one day, the largest gap, and the share of
    calendar days that are missing once reindexed.
    """
    gaps = []
    missing = filled = 0
    for _pid, grp in df.groupby(PATIENT_COL, sort=False):
        dates = pd.DatetimeIndex(grp[DATE_COL].drop_duplicates().sort_values())
        if len(dates) < 2:
            continue
        diffs = dates.to_series().diff().dt.days.dropna().to_numpy()
        gaps.extend(diffs[diffs > 1].tolist())
        span = (dates.max() - dates.min()).days + 1
        filled += len(dates)
        missing += span - len(dates)
    total = filled + missing
    return {
        "n_gap_transitions": int(len(gaps)),
        "max_gap_days": int(max(gaps)) if gaps else 1,
        "missing_day_fraction": round(missing / total, 4) if total else 0.0,
    }


def pooled_indicator_pairs(series_by_patient, lag: int):
    """Pooled (x_t, x_{t+lag}) pairs across patients, dropping any pair that
    straddles a missing day. Used by the calendar-correct pooled ACF so a
    lag of ``k`` is always exactly ``k`` calendar days within one patient.
    """
    a, b = [], []
    for s in series_by_patient.values():
        v = s.to_numpy(dtype=float)
        if len(v) <= lag:
            continue
        x0, x1 = v[:-lag], v[lag:]
        ok = ~np.isnan(x0) & ~np.isnan(x1)
        a.append(x0[ok])
        b.append(x1[ok])
    if not a:
        return np.array([]), np.array([])
    return np.concatenate(a), np.concatenate(b)
