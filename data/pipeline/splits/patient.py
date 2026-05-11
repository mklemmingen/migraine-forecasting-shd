"""
patient.py - Whole-patient holdout split.

Supports any number of folds via the `cutpoints` parameter:
  cutpoints=(0.70, 0.85) → train / val / test  (70/15/15)
  cutpoints=(0.80,)      → train / test         (80/20)
  cutpoints=(0.70,)      → train / test         (70/30)

Patients are sorted by row count then randomly permuted (seed=42). A greedy
walk assigns patients to folds as cumulative row proportion crosses each cutpoint.
"""
from typing import Optional

import numpy as np
import pandas as pd

SPLIT_NAME: str = "patient"


def apply_split(
    df: pd.DataFrame,
    boundaries: Optional[dict] = None,
    cutpoints: tuple = (0.70, 0.85),
    fold_names: tuple = ('train', 'val', 'test'),
    seed: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Add a 'split' column by assigning whole patients to folds.

    First call (diary):
        df_split, boundaries = apply_split(diary_df, cutpoints=(...), fold_names=(...), seed=42)

    Second call (disability):
        dis_split, _ = apply_split(disability_df, boundaries=boundaries)

    When `boundaries` is supplied all other kwargs are ignored.
    """
    if boundaries is None:
        patient_counts = (
            df.groupby('patient_id')
            .size()
            .reset_index(name='n_rows')
            .sort_values(['n_rows', 'patient_id'])
        )
        patients = patient_counts['patient_id'].tolist()
        rng = np.random.default_rng(seed)
        rng.shuffle(patients)

        total_rows = df.shape[0]
        cumulative = 0
        assignment: dict = {}
        count_lookup = patient_counts.set_index('patient_id')['n_rows'].to_dict()

        for pid in patients:
            frac = cumulative / total_rows
            fold_idx = len(cutpoints)  # default to last fold
            for i, cut in enumerate(cutpoints):
                if frac < cut:
                    fold_idx = i
                    break
            assignment[pid] = fold_names[fold_idx]
            cumulative += count_lookup[pid]

        boundaries = {'patient_assignment': assignment, 'fold_names': list(fold_names)}

    assignment = boundaries['patient_assignment']
    fold_names = boundaries['fold_names']
    out = df.copy()
    out['split'] = out['patient_id'].map(assignment).fillna(fold_names[0])
    return out, boundaries
