"""Smoke tests for experiment/_eval/metrics_lib.py.

Run with:
    .venv/bin/python -m pytest tests/test_metrics_lib.py -q

These guard the bootstrap and calibration primitives against silent
regression. The bootstrap path is the single source of headline CIs
in the paper, so a regression here would silently distort every
reported interval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiment"))
from _eval.metrics_lib import (  # noqa: E402
    expected_calibration_error,
    calibration_slope,
    find_operating_thresholds,
    run_bootstrap_evaluation,
)


_NUM = re.compile(r"-?\d+\.\d+")


def _first_number(s: str) -> float:
    m = _NUM.search(s)
    return float(m.group(0)) if m else float("nan")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _toy(rng, n=500, prevalence=0.1, edge=0.3):
    y = rng.binomial(1, prevalence, size=n)
    p = np.clip(rng.beta(2, 5, size=n) + edge * y, 0, 1)
    return y, p


def test_ece_returns_float_in_unit_interval(rng):
    y, p = _toy(rng)
    e = expected_calibration_error(y, p)
    assert isinstance(e, float)
    assert 0.0 <= e <= 1.0


def test_calibration_slope_perfect_classifier_is_finite():
    """A perfectly-calibrated classifier should give a slope near 1."""
    rng = np.random.default_rng(0)
    n = 10_000
    p = rng.uniform(0, 1, size=n)
    y = rng.binomial(1, p)
    s = calibration_slope(y, p)
    assert 0.7 < s < 1.3  # broad band - the toy classifier is approximate


def test_find_operating_thresholds_returns_two_floats(rng):
    y, p = _toy(rng)
    mcc_t, sens_t = find_operating_thresholds(y, p)
    assert 0.0 <= mcc_t <= 1.0
    assert 0.0 <= sens_t <= 1.0


def test_bootstrap_returns_metric_strings(rng):
    """The bootstrap returns one 'mean [lo - hi]' string per metric and
    its AUROC point-estimate matches the analytic value within 2%."""
    y, p = _toy(rng, n=2000)
    y_series = pd.Series(y)
    out = run_bootstrap_evaluation(
        y_series, p, opt_mcc_thresh=0.5, sens_05_thresh=0.5,
        n_iterations=200, seed=42,
    )
    assert "AUROC" in out
    assert "AUPRC" in out
    # The string is "mean [lo - hi]"; the first number is the point estimate.
    auc_boot = _first_number(out["AUROC"])
    auc_an = roc_auc_score(y, p)
    assert abs(auc_boot - auc_an) < 0.02


def test_bootstrap_seed_determinism(rng):
    """Two runs at the same seed must produce identical metric strings."""
    y, p = _toy(rng, n=2000)
    y_series = pd.Series(y)
    out1 = run_bootstrap_evaluation(y_series, p, 0.5, 0.5, n_iterations=200, seed=42)
    out2 = run_bootstrap_evaluation(y_series, p, 0.5, 0.5, n_iterations=200, seed=42)
    assert out1 == out2


def test_bootstrap_different_seeds_diverge(rng):
    """Two runs at different seeds usually produce different AUROC point estimates."""
    y, p = _toy(rng, n=2000)
    y_series = pd.Series(y)
    out1 = run_bootstrap_evaluation(y_series, p, 0.5, 0.5, n_iterations=200, seed=42)
    out2 = run_bootstrap_evaluation(y_series, p, 0.5, 0.5, n_iterations=200, seed=43)
    # The string of every metric should differ in at least one number, given
    # different bootstrap samples drawn.
    assert out1["AUROC"] != out2["AUROC"]
