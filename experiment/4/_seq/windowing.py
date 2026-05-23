"""Gap-aware, within-split look-back windowing for Addition 4.

For each labelled diary row, build a fixed-length look-back window of the
preceding days' engineered features, so a sequence model can consume the
recent per-patient history. The contract is row-aligned: ``make_windows``
returns exactly one window (and one mask) per input row, so downstream
predictions line up 1:1 with the rows the leaf evaluator scores - this is
what lets experiment/4's evaluate.py stay identical to Additions 0/1.

Two disciplines, both anchored in the project's own findings:

* Leakage. Windowing runs WITHIN a single split frame (train, val, or
  test); a row never draws look-back days from another split. This is the
  row-level form of the feature-channel leakage in
  docs/results_findings.md Section 1, and avoids the
  sequence-construction-before-split failure mode quantified by Albelali &
  Ahmed (arXiv:2512.06932, p. 2). Because make_windows is always called on
  one split frame at a time, the discipline is structural, not optional.

* Gaps. The diary has 208 transitions with gaps > 1 day, max 37
  (docs/addition3_results.md). A window must not recur across a long gap as
  if the days were contiguous. Days separated by more than
  ``gap_segment_days`` calendar days from the target row are masked out of
  its window.

The companion calendar-regular series builder for the descriptive temporal
analysis lives at experiment/3/_temporal/series.py; this module is its
forecasting analogue over the full engineered feature matrix rather than the
single binary attack indicator.
"""
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd

PATIENT_COL = "patient_id"
DATE_COL = "date"
ID_COLS = (PATIENT_COL, DATE_COL, "entry_id")


class WindowBatch(NamedTuple):
    """Row-aligned windowed batch.

    sequences : float array (n_rows, lookback, n_features)
        Per-row look-back window, oldest day first, newest day last. Masked
        positions are filled with 0.0 (the model must respect ``mask``).
    mask      : float array (n_rows, lookback)
        1.0 where the timestep is a real observed day inside the gap-segment,
        0.0 where it is padding or beyond a gap. Feed to the model so padded
        steps do not contribute.
    row_index : int array (n_rows,)
        Positional index into the input frame for each window, so callers can
        scatter per-row predictions back to the original row order.
    n_features : int
    """
    sequences: np.ndarray
    mask: np.ndarray
    row_index: np.ndarray
    n_features: int


def feature_columns(X: pd.DataFrame) -> List[str]:
    """Feature columns of an id-bearing X (everything except the id columns)."""
    return [c for c in X.columns if c not in ID_COLS]


def _patient_blocks(X: pd.DataFrame):
    """Yield (positional_indices, dates, feature_matrix) per patient, in date
    order. Positional indices are into the original (pre-sort) frame so the
    caller can realign predictions to input rows.
    """
    feats = feature_columns(X)
    pos = np.arange(len(X))
    dates = pd.to_datetime(X[DATE_COL]).to_numpy()
    fmat = X[feats].to_numpy(dtype=float)
    for _pid, idx in X.groupby(PATIENT_COL, sort=False).indices.items():
        order = idx[np.argsort(dates[idx])]
        yield pos[order], dates[order], fmat[order]


def make_windows(
    X: pd.DataFrame,
    *,
    lookback: int,
    gap_segment_days: int = 7,
    feature_cols: Optional[List[str]] = None,
) -> WindowBatch:
    """Build one gap-aware look-back window per row of an id-bearing X.

    Parameters mirror the SequenceClassifier hyper-parameters. ``feature_cols``
    is accepted so the estimator can lock the train-time column order and
    reuse it at predict time (guarding against column reordering between
    splits); when None it is derived from X.

    Returns a WindowBatch whose first axis is aligned to ``X`` row order.

    Edge policy (decisions in docs/addition4_sequence.md Section 9):
      * Left-pad + mask (Decision 1). Each window ends at the current day,
        right-aligned, so the last timestep is ALWAYS the observed current day;
        the start of a patient's series is left-padded with zeros and mask=0.
        Every input row is therefore predictable and the output stays 1:1 with
        X - which is what lets predict_proba return (n, 2) for all rows.
      * Gap channel (Decision 1, after Che et al. 2018, GRU-D). A final feature
        channel carries the time interval since the previous retained day,
        normalised to [0, 1] by ``gap_segment_days`` - the GRU-D "time interval"
        representation of informative missingness, so the model sees calendar
        gaps even though the window indexes recorded days. n_features therefore
        equals len(features) + 1.
      * Gap segmentation (Decision 2). Stepping back from the current day stops
        as soon as two consecutive recorded days differ by more than
        ``gap_segment_days`` calendar days, so a window never bridges a long
        recording gap (docs/addition3_results.md: ACF decays by day 3-7).
    """
    feats = feature_cols if feature_cols is not None else feature_columns(X)
    n_base = len(feats)
    n_features = n_base + 1  # + per-step time-interval (gap) channel
    Xf = X[[PATIENT_COL, DATE_COL] + feats]

    sequences = np.zeros((len(X), lookback, n_features), dtype=np.float32)
    mask = np.zeros((len(X), lookback), dtype=np.float32)
    row_index = np.arange(len(X))

    def _days(d_to, d_from) -> float:
        return float((d_to - d_from) / np.timedelta64(1, "D"))

    for pos, dates, fmat in _patient_blocks(Xf):
        for t in range(len(pos)):
            # Walk back from the current day t, newest-first, stopping at a gap.
            retained = [t]
            k = t
            while len(retained) < lookback and k - 1 >= 0:
                if _days(dates[k], dates[k - 1]) > gap_segment_days:
                    break
                retained.append(k - 1)
                k -= 1
            retained.reverse()  # oldest-first
            start = lookback - len(retained)  # right-align (last step = day t)
            row = pos[t]
            for s, day in enumerate(retained):
                sequences[row, start + s, :n_base] = fmat[day]
                gap = 0.0 if s == 0 else _days(dates[day], dates[retained[s - 1]])
                sequences[row, start + s, n_base] = min(gap, gap_segment_days) / gap_segment_days
                mask[row, start + s] = 1.0

    return WindowBatch(sequences, mask, row_index, n_features)
