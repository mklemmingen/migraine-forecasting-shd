"""load_seq - split loader that keeps patient_id and date on X.

Additions 0/1 call ``_dataRead.read.prep_split``, which drops patient_id,
date, entry_id, the target, and cv_fold before X reaches the model. A
sequence model needs patient_id and date to build per-patient, calendar-
gap-aware, chronologically-ordered windows, so this loader keeps those two
columns on X and drops only the genuinely non-predictive ones. The
SequenceClassifier (sklearn_wrapper.py) separates the id columns from the
feature columns at fit/predict time via _seq.windowing.feature_columns.

The feature-set filters (no_rolling_features, park_features) plug in via the
same ``loader`` callable contract as _dataRead.read.load_raw, so Addition 4
reuses the existing column whitelists unchanged.
"""
from typing import Callable, Optional, Tuple

import pandas as pd

# Resolved against experiment/ on sys.path by the leaf scripts.
from _dataRead.read import load_raw, TARGET_COL  # noqa: E402

# Columns dropped from X by the sequence loader. patient_id and date are
# deliberately NOT dropped here (the windower needs them); entry_id and
# cv_fold are never predictors.
SEQ_DROP_COLS: Tuple[str, ...] = ("entry_id", "cv_fold", TARGET_COL)


def seq_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split an in-memory DataFrame into (X_with_ids, y) for the windower.

    Like _dataRead.read.prep_split but keeps patient_id and date on X (drops
    only entry_id, cv_fold, and the target). The CV evaluator calls this on
    per-fold sub-frames it has already carved with the cv_fold/date columns,
    so the windower still receives the id columns it needs.
    """
    X = df.drop(columns=[c for c in SEQ_DROP_COLS if c in df.columns])
    y = df[TARGET_COL]
    return X, y


def load_seq(
    filepath: str,
    loader: Optional[Callable[[str], pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a split parquet and return (X_with_ids, y).

    X retains patient_id and date alongside the feature columns; y is the
    binary ``migraine_target`` Series. Pass ``loader`` (e.g.
    ``select_non_rolling_features``) to apply a feature-set whitelist before
    the id/target split, exactly as in _dataRead.read.
    """
    return seq_split(load_raw(filepath, loader=loader))
