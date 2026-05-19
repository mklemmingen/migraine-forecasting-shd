"""
read.py - Shared data-loading utilities for addition leaf scripts.

Centralises (X, y) extraction so train/evaluate/evaluate_cv files don't
duplicate column-drop logic. Custom loaders (e.g. spano-feature filtering)
plug in via the `loader` callable parameter.

Public API:
    prep_split(df)                            -> (X, y)
    load_and_prep_data(filepath, loader=None) -> (X, y)
    load_raw(filepath, loader=None)           -> raw DataFrame with date column
    chronological_subsplit(train_fold, cal_ratio=0.20) -> (train_sub, cal_sub)

``load_raw`` is the entry point when the caller needs the raw DataFrame
including the date column (e.g. to feed chronological_subsplit before
prep_split drops the date). Keeping pandas behind this helper means
leaf scripts never need to import pandas themselves and avoids the
"forgot to import pd" failure mode in templates that substitute a
``pd.foo`` call into their generated code.

Usage from a leaf script (depth-agnostic - walks up to experiment/):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(next(
        p for p in Path(__file__).resolve().parents if p.name == 'experiment')))
    from _dataRead.read import load_and_prep_data, prep_split
"""
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

ID_COLS: Tuple[str, ...] = ('entry_id', 'patient_id', 'date')
TARGET_COL: str = 'migraine_target'
NON_FEATURE_COLS: Tuple[str, ...] = ID_COLS + (TARGET_COL, 'cv_fold')


def prep_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop identifier and label columns from a DataFrame; return (X, y).

    Tolerates absence of cv_fold so the same call works for both
    70/15/15 split parquets and diary_cv5_timeseries.parquet.
    """
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    y = df[TARGET_COL]
    return X, y


def load_and_prep_data(
    filepath: str,
    loader: Optional[Callable[[str], pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a parquet (or apply a custom loader) and return (X, y).

    Pass loader=remove_non_spano_features for the spano-features variants.
    Default loader is pd.read_parquet.
    """
    return prep_split(load_raw(filepath, loader=loader))


def load_raw(
    filepath: str,
    loader: Optional[Callable[[str], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Load a parquet (or apply a custom loader) and return the raw DataFrame.

    Use this when the date column is needed downstream
    (chronological_subsplit reads it before prep_split drops it). Default
    loader is pd.read_parquet. Custom loaders are the column-filtering
    wrappers like select_spano_features, select_park_features.
    """
    return loader(filepath) if loader is not None else pd.read_parquet(filepath)


def chronological_subsplit(
    train_fold: pd.DataFrame,
    cal_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a CV training fold chronologically into (train_sub, cal_sub).

    cal_sub holds the last cal_ratio of unique dates. Used by CV evaluators
    to fit calibrators and select operating thresholds without ever
    touching the evaluation fold.
    """
    unique_dates = np.sort(train_fold['date'].unique())
    cutoff_idx = int(len(unique_dates) * (1.0 - cal_ratio))
    cutoff_date = unique_dates[cutoff_idx]
    train_sub = train_fold[train_fold['date'] < cutoff_date]
    cal_sub = train_fold[train_fold['date'] >= cutoff_date]
    return train_sub, cal_sub
