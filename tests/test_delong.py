"""Smoke tests for experiment/_eval/delong.py.

Run with:
    .venv/bin/python -m pytest tests/test_delong.py -q

These tests defend the DeLong implementation against silent regression.
They do not validate the algorithm against a published reference (the
embedded smoke check in delong.py:__main__ does that against
sklearn.metrics.roc_auc_score for the point estimate). They check
shape invariants, edge cases, and basic monotonicity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment"))
from _eval.delong import delong_auc_ci, delong_paired_test  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_pair(rng, n, prevalence, edge_a, edge_b):
    y = rng.binomial(1, prevalence, size=n)
    a = rng.beta(2, 5, size=n) + edge_a * y
    b = rng.beta(2, 5, size=n) + edge_b * y
    return y, a, b


def test_point_estimate_matches_sklearn(rng):
    y, a, _ = _make_pair(rng, 500, 0.1, 0.4, 0.2)
    auc, _, _ = delong_auc_ci(y, a)
    assert abs(auc - roc_auc_score(y, a)) < 1e-9


def test_ci_brackets_point_estimate(rng):
    y, a, _ = _make_pair(rng, 500, 0.1, 0.4, 0.2)
    auc, lo, hi = delong_auc_ci(y, a)
    assert lo <= auc <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_ci_width_shrinks_with_n(rng):
    """Doubling sample size should reduce the CI width by roughly sqrt(2)."""
    y1, a1, _ = _make_pair(rng, 500, 0.1, 0.4, 0.2)
    y2, a2, _ = _make_pair(rng, 5000, 0.1, 0.4, 0.2)
    _, lo1, hi1 = delong_auc_ci(y1, a1)
    _, lo2, hi2 = delong_auc_ci(y2, a2)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_paired_test_detects_real_difference(rng):
    """Two classifiers with clearly different AUCs should yield p < 0.01."""
    y, a, b = _make_pair(rng, 1000, 0.1, 0.5, 0.1)
    auc_a, auc_b, delta, p, lo, hi = delong_paired_test(y, a, b)
    assert auc_a > auc_b
    assert delta > 0
    assert p < 0.01
    assert lo > 0      # CI of the diff is strictly above zero
    assert lo <= delta <= hi


def test_paired_test_zero_difference_gives_high_p(rng):
    """Two copies of the same scores -> p value should be ~1.0 and delta ~0."""
    y, a, _ = _make_pair(rng, 500, 0.1, 0.3, 0.3)
    auc_a, auc_b, delta, p, _, _ = delong_paired_test(y, a, a)
    assert abs(delta) < 1e-12
    assert p == 1.0


def test_raises_on_no_positives():
    y = np.zeros(20, dtype=int)
    score = np.arange(20, dtype=float)
    with pytest.raises(ValueError):
        delong_auc_ci(y, score)


def test_raises_on_non_binary():
    y = np.array([0, 1, 2, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError):
        delong_auc_ci(y, score)


def test_handles_ties_via_midrank(rng):
    """Tied scores should not crash and should give a valid AUC."""
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.5, 0.5, 0.5, 0.8, 0.2, 0.8])  # ties throughout
    auc, lo, hi = delong_auc_ci(y, score)
    assert 0.0 <= lo <= auc <= hi <= 1.0
    # sklearn also handles ties; match to 4 decimals
    assert abs(auc - roc_auc_score(y, score)) < 1e-4
