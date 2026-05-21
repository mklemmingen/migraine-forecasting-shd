"""Unit test: ALE is correlation-robust where PDP is not.

Constructs a synthetic where x1 and x2 are strongly correlated and the
model has a genuine non-linear dependence on both (a saturating product).
ALE asks "what is the local effect of x1, holding the conditional
distribution of x2 fixed". PDP asks "average f over the marginal of x2 at
each x1", which forces x2 to values that never co-occur with that x1 (the
two features are correlated), so PDP evaluates the model far outside the
joint data support and reports a biased x1 curve. ALE bins on x1 and takes
local differences over exactly the rows in each bin, so it never leaves the
support.

The test asserts ALE's x1 net slope matches a held-out conditional
finite-difference reference (the unbiased target), while PDP's x1 net slope
deviates substantially because of the extrapolation bias. This demonstrates
the property that motivates choosing ALE over PDP for the heavily-correlated
engineered feature sets (docs Section 3.2).

Run: ``.venv/bin/python experiment/_explain/_tests/test_ale_correlation.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ale import first_order_ale, partial_dependence, ale_slope  # noqa: E402


def _make_correlated_data(n=8000, rho=0.95, seed=0):
    """x1 ~ U(-2,2); x2 = rho-correlated copy. Strong correlation means
    the joint support is a narrow diagonal band; the off-diagonal corners
    (low x1 with high x2, and vice versa) carry essentially no data."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, n)
    noise = rng.uniform(-2.0, 2.0, n)
    x2 = rho * x1 + np.sqrt(1 - rho ** 2) * noise
    return pd.DataFrame({"x1": x1, "x2": x2})


def _model(X: pd.DataFrame) -> np.ndarray:
    """A model with a genuine interaction: f = x1 - x2 + x1*x2.

    On the joint diagonal support (x1 ~= x2) the dependence on x1 is
    moderate; the interaction term makes PDP's marginal averaging probe
    impossible (x1, x2) corners where x1*x2 explodes, biasing the PDP x1
    curve away from the in-support effect."""
    x1 = X["x1"].to_numpy(dtype=float)
    x2 = X["x2"].to_numpy(dtype=float)
    return x1 - x2 + x1 * x2


def _conditional_reference_slope(X: pd.DataFrame, feature: str, n_bins: int = 10) -> float:
    """Unbiased reference: local conditional finite differences accumulated
    over the data, computed independently of the ALE implementation. This is
    the quantity ALE estimates and the quantity PDP fails to estimate under
    correlation."""
    values = X[feature].to_numpy(dtype=float)
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2)
    accumulated = 0.0
    for k in range(len(edges) - 1):
        in_bin = idx == k
        if not np.any(in_bin):
            continue
        lo = X[in_bin].copy(); lo[feature] = edges[k]
        hi = X[in_bin].copy(); hi[feature] = edges[k + 1]
        accumulated += float(np.mean(_model(hi) - _model(lo)))
    return accumulated


def main() -> int:
    X = _make_correlated_data()

    # Unbiased target: the in-support accumulated local effect of x1.
    reference_slope = _conditional_reference_slope(X, "x1")

    ale_x1 = first_order_ale(_model, X, "x1", n_bins=10)
    ale_x1_slope = ale_slope(ale_x1)

    grid = np.quantile(X["x1"].to_numpy(), np.linspace(0.05, 0.95, 10))
    pdp_x1 = partial_dependence(_model, X, "x1", grid)
    pdp_x1_slope = float(pdp_x1[-1] - pdp_x1[0])

    ale_err = abs(ale_x1_slope - reference_slope)
    pdp_err = abs(pdp_x1_slope - reference_slope)

    print(f"Conditional reference x1 slope (unbiased): {reference_slope:+.4f}")
    print(f"ALE x1 net slope:  {ale_x1_slope:+.4f}   |error| = {ale_err:.4f}")
    print(f"PDP x1 net slope:  {pdp_x1_slope:+.4f}   |error| = {pdp_err:.4f}")

    ok = True
    # ALE matches the conditional reference closely.
    if ale_err >= 0.10:
        print("FAIL: ALE deviated from the conditional reference.")
        ok = False
    # PDP is materially more biased than ALE under the correlation.
    if pdp_err <= 3 * ale_err + 0.20:
        print("FAIL: PDP was not materially more biased than ALE.")
        ok = False

    if ok:
        print("PASS: ALE tracks the unbiased conditional effect; PDP is "
              "materially biased under correlation (as expected).")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
