"""
cv_folds.py - Expanding-window time-series CV fold labels.

Why CV instead of a fixed val split:
With n_test=136 and 19 positives the 70/15/15 test set is too small to reliably rank models
(95% CI on AUROC spans ~0.3).

Using 5-fold time-series CV on the full dataset averages evaluation over 5 val windows (~830 rows each),
possibly reducing variance in model comparison.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def assign_cv_folds(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Assign expanding-window CV fold labels to the engineered DataFrame.

    Operates on unique chronological dates so that every row belonging to the
    same date lands in the same fold - consistent with the date-based splits.

    cv_fold=0  - always train (dates before the first val window)
    cv_fold=k  - validation data for fold k (k = 1..n_splits)

    Usage in experiment scripts:
        cv = pd.read_parquet('diary_cv5_timeseries.parquet')
        for fold in range(1, n_splits + 1):
            train = cv[cv['cv_fold'] < fold]   # expanding window
            val   = cv[cv['cv_fold'] == fold]
    """
    unique_dates = np.sort(df['date'].unique())
    tss = TimeSeriesSplit(n_splits=n_splits)

    date_to_fold = {d: 0 for d in unique_dates}
    for fold_idx, (_, val_indices) in enumerate(tss.split(unique_dates), start=1):
        for i in val_indices:
            date_to_fold[unique_dates[i]] = fold_idx

    out = df.copy()
    out['cv_fold'] = out['date'].map(date_to_fold).astype(int)
    return out
