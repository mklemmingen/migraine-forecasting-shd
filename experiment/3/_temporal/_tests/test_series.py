"""Unit tests for the gap-aware calendar-regular series.

Verifies that a known multi-day gap becomes explicit NaN calendar days, so
a lag of k is k calendar days rather than k records.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import series as S  # noqa: E402


def _synthetic():
    # One patient: day 0 attack, day 1 attack, then a 37-day gap, then a day.
    base = pd.Timestamp("2014-01-01")
    rows = [
        ("p1", base,                       1.0),
        ("p1", base + pd.Timedelta("1D"),  1.0),
        ("p1", base + pd.Timedelta("38D"), 0.0),   # 37-day gap after day 1
    ]
    return pd.DataFrame(rows, columns=["patient_id", "date", "migraine_target"])


def test_reindex_inserts_calendar_gap():
    df = _synthetic()
    sbp = S.build_patient_series(df, "migraine_target")
    s = sbp["p1"]
    # Span is 39 calendar days (day 0..38 inclusive).
    assert len(s) == 39, len(s)
    # The two recorded attacks are at index 0 and 1, the last record at 38.
    assert s.iloc[0] == 1.0 and s.iloc[1] == 1.0 and s.iloc[38] == 0.0
    # Everything between is missing (NaN), not zero.
    assert s.iloc[2:38].isna().all()


def test_pooled_pairs_respect_calendar_lag():
    df = _synthetic()
    sbp = S.build_patient_series(df, "migraine_target")
    # Lag 1: only the (day0, day1) pair survives; the gap days are NaN.
    a, b = S.pooled_indicator_pairs(sbp, lag=1)
    assert list(a) == [1.0] and list(b) == [1.0], (a, b)
    # Lag 37: pairs day1 (idx1) with idx38, both observed -> one pair.
    a37, b37 = S.pooled_indicator_pairs(sbp, lag=37)
    assert list(a37) == [1.0] and list(b37) == [0.0], (a37, b37)


def test_gap_summary_counts_the_gap():
    g = S.gap_summary(_synthetic())
    assert g["n_gap_transitions"] == 1 and g["max_gap_days"] == 37, g


if __name__ == "__main__":
    test_reindex_inserts_calendar_gap()
    test_pooled_pairs_respect_calendar_lag()
    test_gap_summary_counts_the_gap()
    print("series tests passed")
