"""
stratified.py — Class-balanced random shuffle split.

Supports any number of folds via the `cutpoints` parameter:
  cutpoints=(0.70, 0.85) → train / val / test  (70/15/15)
  cutpoints=(0.80,)      → train / test         (80/20)
  cutpoints=(0.70,)      → train / test         (70/30)

Rows are shuffled while preserving class proportions in each fold via
stratified splitting. This breaks temporal ordering intentionally.
# See data/processed/dataset_characterization.pdf for current positive rate.
"""
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAME: str = "stratified"


def apply_split(
    df: pd.DataFrame,
    boundaries: Optional[dict] = None,
    cutpoints: tuple = (0.70, 0.85),
    fold_names: tuple = ('train', 'val', 'test'),
    seed: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Add a 'split' column using class-balanced random assignment.

    First call (diary — requires 'migraine_target' for stratification):
        df_split, boundaries = apply_split(diary_df, cutpoints=(...), fold_names=(...), seed=42)

    Second call (disability — matched via (patient_id, date) keys):
        dis_split, _ = apply_split(disability_df, boundaries=boundaries)

    When `boundaries` is supplied all other kwargs are ignored.
    """
    if boundaries is None:
        idx = df.index
        y = df['migraine_target']
        fold_assignment: dict = {}

        if len(cutpoints) == 1:
            # Two-way split
            train_idx, other_idx = train_test_split(
                idx, test_size=1.0 - cutpoints[0], stratify=y, random_state=seed)
            _record(df, train_idx, fold_names[0], fold_assignment)
            _record(df, other_idx, fold_names[1], fold_assignment)

        else:
            # Multi-way: peel off the last fold first, then recurse inward
            test_size = 1.0 - cutpoints[-1]
            train_val_idx, test_idx = train_test_split(
                idx, test_size=test_size, stratify=y, random_state=seed)
            _record(df, test_idx, fold_names[-1], fold_assignment)

            # For the 3-way case: val is (cutpoints[-1] - cutpoints[-2]) / cutpoints[-1]
            val_size = (cutpoints[-1] - cutpoints[-2]) / cutpoints[-1]
            inner_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_size,
                stratify=y.loc[train_val_idx], random_state=seed)
            _record(df, val_idx, fold_names[-2], fold_assignment)
            _record(df, inner_idx, fold_names[0], fold_assignment)

        boundaries = {'date_assignment': fold_assignment, 'fold_names': list(fold_names)}

    date_assignment = boundaries['date_assignment']
    fold_names = boundaries['fold_names']
    out = df.copy()
    out['split'] = out.apply(
        lambda r: date_assignment.get((r['patient_id'], r['date']), fold_names[0]),
        axis=1,
    )
    return out, boundaries


def _record(df: pd.DataFrame, idx, fold_name: str, d: dict) -> None:
    """Store (patient_id, date) → fold_name for each row in idx."""
    rows = df.loc[idx, ['patient_id', 'date']]
    for _, row in rows.iterrows():
        d[(row['patient_id'], row['date'])] = fold_name
