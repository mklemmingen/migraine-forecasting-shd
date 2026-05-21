"""Accumulated Local Effects (ALE) via conditional-bin accumulation.

ALE [apley2020ale] estimates a feature's effect by averaging *local*
prediction differences within narrow bins of the feature, accumulated
across bins. Because the differences are taken over the conditional
distribution inside each bin (the rows that actually have that feature
range), ALE stays unbiased when features are correlated. Partial
dependence, by contrast, averages predictions over the marginal
distribution, which fabricates physically impossible feature
combinations when features are correlated and biases the estimated
effect [apley2020ale, Sec. 1]. The engineered feature sets here are
heavily correlated by construction (rolling/lag/streak features are
deterministic functions of the same-day flags), so ALE is the primary
effect plot and PDP is shown only as an agreement overlay.

First-order (main-effect) ALE only; this module computes the centred
accumulated effect curve and exposes the per-bin local differences for
the PDP-agreement check.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Quantile bins keep roughly equal mass per bin so sparse tails do not
# dominate the accumulated effect. Binary/low-cardinality features fall
# back to their distinct values.
_N_BINS = 10


def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges; collapses to the distinct values for
    low-cardinality (e.g. binary) features."""
    distinct = np.unique(values)
    if len(distinct) <= n_bins + 1:
        return distinct.astype(float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)
    return np.unique(edges)


def first_order_ale(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature: str,
    n_bins: int = _N_BINS,
) -> dict:
    """Compute the centred first-order ALE curve for ``feature``.

    For each bin ``(z_{k-1}, z_k]`` the rows whose feature value falls in
    the bin are pushed to both edges; the model-prediction difference
    ``f(z_k) - f(z_{k-1})`` is averaged over exactly those rows
    (conditional, not marginal). The bin-mean differences are accumulated
    and mean-centred so the curve integrates to zero over the data.

    Returns ``{x, ale, bin_local_diff, n_bins}`` where ``x`` are bin
    centres and ``ale`` the centred accumulated effect at each centre.
    """
    values = X[feature].to_numpy(dtype=float)
    edges = _bin_edges(values, n_bins)
    if len(edges) < 2:
        return {"x": edges, "ale": np.zeros_like(edges), "bin_local_diff": np.array([]), "n_bins": 0}

    # Assign each row to a bin index in [0, len(edges)-2].
    idx = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2)

    local_diff = np.zeros(len(edges) - 1)
    bin_counts = np.zeros(len(edges) - 1)
    for k in range(len(edges) - 1):
        in_bin = idx == k
        if not np.any(in_bin):
            continue
        X_lo = X[in_bin].copy()
        X_hi = X[in_bin].copy()
        X_lo[feature] = edges[k]
        X_hi[feature] = edges[k + 1]
        diff = predict_fn(X_hi) - predict_fn(X_lo)
        local_diff[k] = float(np.mean(diff))
        bin_counts[k] = int(in_bin.sum())

    accumulated = np.concatenate([[0.0], np.cumsum(local_diff)])
    # Mean-centre over the data so the curve has zero mean weighted by bin mass.
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = bin_counts.sum()
    if total > 0:
        weighted_mean = np.sum(0.5 * (accumulated[:-1] + accumulated[1:]) * bin_counts) / total
    else:
        weighted_mean = 0.0
    ale = accumulated - weighted_mean
    return {
        "x": edges,
        "ale": ale,
        "bin_local_diff": local_diff,
        "bin_counts": bin_counts,
        "n_bins": int(len(edges) - 1),
    }


def partial_dependence(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    feature: str,
    grid: np.ndarray,
) -> np.ndarray:
    """Marginal partial dependence on ``grid`` for the agreement overlay.

    Shown only as a "no correlation artefact here" signal where it tracks
    ALE; where they diverge the divergence is reported and ALE is trusted.
    """
    pd_vals = np.zeros(len(grid))
    for i, g in enumerate(grid):
        Xg = X.copy()
        Xg[feature] = g
        pd_vals[i] = float(np.mean(predict_fn(Xg)))
    return pd_vals - pd_vals.mean()


def ale_slope(curve: dict) -> float:
    """Net signed slope of the ALE curve (last minus first centred value),
    a compact direction-and-magnitude summary for the per-leaf text report."""
    ale = curve["ale"]
    if len(ale) < 2:
        return 0.0
    return float(ale[-1] - ale[0])
