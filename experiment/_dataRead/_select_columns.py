"""_select_columns.py - Inclusion-semantics column selection helper.

Backs the ``filter_to_*`` modules under ``_dataRead/``. Each filter
declares a positive whitelist (the columns it expects to find in the
upstream engineered parquet) instead of an exclusion list, so any
new column added to the engineering pipeline is silently dropped by
the filter rather than leaking through to the model. If a whitelisted
column is missing from the parquet, ``select_columns`` raises a clear
error so the feature-set definition fails loudly when the schema
drifts.
"""
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def select_columns(
    parquet_path: str | Path,
    *,
    required: Sequence[str],
    optional: Iterable[str] = (),
) -> pd.DataFrame:
    """Read a parquet and return only the named columns.

    Args:
        parquet_path: Path to the input parquet (``str`` or ``Path``).
        required: Column names that must be present in the parquet.
            Any missing name raises ``ValueError`` with a diagnostic
            listing all available columns - the schema has drifted and
            the caller's whitelist needs updating.
        optional: Column names that pass through if present, are
            silently skipped if absent. Useful for identifier columns
            that exist on some split parquets but not on others (e.g.
            ``cv_fold`` only on CV-split parquets).

    Returns:
        DataFrame containing exactly ``required + (optional ∩ available)``
        columns, in the order they appear in the parquet's own column
        index (for stable downstream behaviour).
    """
    path = Path(parquet_path)
    df = pd.read_parquet(path)

    required_list = list(required)
    missing = [c for c in required_list if c not in df.columns]
    if missing:
        raise ValueError(
            f"select_columns: required columns missing from {path.name}: "
            f"{missing!r}. Available columns: {sorted(df.columns.tolist())!r}"
        )

    optional_present = [c for c in optional if c in df.columns]
    keep = set(required_list) | set(optional_present)

    # Preserve the parquet's own column ordering for stable behaviour
    # across different writes; pandas is column-order sensitive when
    # downstream code uses positional access.
    ordered = [c for c in df.columns if c in keep]
    return df[ordered].copy()
