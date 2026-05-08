"""
chrono.py — Chronological split on global date percentiles.

Supports any number of folds via the `cutpoints` parameter:
  cutpoints=(0.70, 0.85) → train / val / test  (70/15/15)
  cutpoints=(0.80,)      → train / test         (80/20)
  cutpoints=(0.70,)      → train / test         (70/30)
"""
from typing import Optional

import numpy as np
import pandas as pd

SPLIT_NAME: str = "chrono"


def apply_split(
    df: pd.DataFrame,
    boundaries: Optional[dict] = None,
    cutpoints: tuple = (0.70, 0.85),
    fold_names: tuple = ('train', 'val', 'test'),
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Add a 'split' column using global date-percentile boundaries.

    First call (diary):
        df_split, boundaries = apply_split(diary_df, cutpoints=(...), fold_names=(...))

    Second call (disability — reuses diary boundaries):
        dis_split, _ = apply_split(disability_df, boundaries=boundaries)

    When `boundaries` is supplied, `cutpoints` and `fold_names` are ignored —
    the values stored in the boundaries dict are used instead.
    """
    if boundaries is None:
        unique_dates = np.sort(df['date'].unique())
        n = len(unique_dates)
        date_cuts = [unique_dates[int(n * c)] for c in cutpoints]
        boundaries = {'date_cuts': date_cuts, 'fold_names': list(fold_names)}

    date_cuts = boundaries['date_cuts']
    fold_names = boundaries['fold_names']

    out = df.copy()
    conditions = []
    choices = []
    for i, cut in enumerate(date_cuts):
        cond = (out['date'] < cut) if i == 0 else (out['date'] >= date_cuts[i - 1]) & (out['date'] < cut)
        conditions.append(cond)
        choices.append(fold_names[i])

    out['split'] = np.select(conditions, choices, default=fold_names[-1])
    return out, boundaries
