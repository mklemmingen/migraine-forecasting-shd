"""DeLong AUROC confidence intervals and paired AUROC comparison.

Companion to the existing bootstrap-percentile CIs in metrics_lib. The
DeLong method is the published closed-form parametric CI for AUROC
[delong1988comparing] and the canonical paired test for comparing two
AUROCs computed on the same labels and risk-set. It uses the Mann-Whitney
U-statistic representation of AUROC plus its asymptotic variance via the
Sun and Xu fast midrank algorithm [sun2014fast], which evaluates V10 and
V01 (the structural components) in O((m+n) log(m+n)) rather than O(mn).

Public API
----------
delong_auc_ci(y, score, alpha=0.05)
    Returns (auc, ci_lo, ci_hi) for a single classifier's predictions.

delong_paired_test(y, score_a, score_b, alpha=0.05)
    Returns (auc_a, auc_b, delta, p_value, delta_lo, delta_hi) for the
    paired difference between two classifiers' AUROCs on the same y.

Both functions accept y as binary {0,1} (or {True,False}) and score as
any monotonic-in-positive-class real-valued array. They are safe against
ties (handled via midrank).

The Sun-Xu midrank machinery is from the public reference implementation
of Yang et al. (commit notes in the repository); equations are also
verifiable against DeLong et al. 1988 Eqs. 4-6.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Return the midrank vector of x with ties averaged.

    Midrank is 1-based: the smallest value has rank 1; tied values share
    the average of the positions they jointly occupy.
    """
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def _delong_components(y: np.ndarray, score: np.ndarray):
    """Return (auc, V10, V01) where V10 is the per-positive structural
    component and V01 the per-negative one.

    Mann-Whitney representation: AUC = (Tx - m*(m+1)/2) / (m*n) where Tx
    is the sum of midranks of the positive scores within the combined
    sample. V10 and V01 are unbiased estimators of the structural
    components used by DeLong's variance formula.
    """
    y = np.asarray(y).astype(int).ravel()
    score = np.asarray(score, dtype=float).ravel()
    if y.shape != score.shape:
        raise ValueError("y and score must have the same shape")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("y must be binary {0, 1}")

    pos = score[y == 1]
    neg = score[y == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        raise ValueError("Need at least one positive and one negative for AUC")

    # Midranks within the combined sample, and within the per-class samples,
    # following Sun-Xu Eqs. 7-8.
    tz = _compute_midrank(np.concatenate([pos, neg]))
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz_pos = tz[:m]
    tz_neg = tz[m:]

    auc = (tz_pos.sum() / m - (m + 1) / 2.0) / n

    # V10[i] = (1 - (rank_within_pos[i] - rank_within_combined[i] + 1 + (m-1)/2) / n) ... rearranged
    # The canonical form: V10 = (tz_pos - tx) / n  ;  V01 = 1 - (tz_neg - ty) / m
    V10 = (tz_pos - tx) / n
    V01 = 1.0 - (tz_neg - ty) / m
    return auc, V10, V01


def delong_auc_ci(y, score, alpha: float = 0.05):
    """DeLong-method 100*(1 - alpha)% CI for a single AUROC.

    Parameters
    ----------
    y      : array of binary labels, shape (N,)
    score  : array of model scores (higher = more positive), shape (N,)
    alpha  : two-sided significance level (default 0.05 -> 95% CI)

    Returns
    -------
    (auc, lo, hi) : tuple of floats; lo and hi are clipped to [0, 1].
    """
    auc, V10, V01 = _delong_components(np.asarray(y), np.asarray(score))
    m = len(V10)
    n = len(V01)
    var = V10.var(ddof=1) / m + V01.var(ddof=1) / n
    se = float(np.sqrt(var))
    z = float(stats.norm.ppf(1 - alpha / 2))
    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    return float(auc), lo, hi


def delong_paired_test(y, score_a, score_b, alpha: float = 0.05):
    """Paired DeLong test for AUROC_a - AUROC_b on the same labels y.

    Returns (auc_a, auc_b, delta, p_value, delta_lo, delta_hi). The CI
    is for the paired difference and accounts for the covariance between
    the two classifiers' Mann-Whitney statistics.
    """
    y = np.asarray(y)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if a.shape != b.shape or a.shape != y.shape:
        raise ValueError("y, score_a, score_b must have the same shape")

    auc_a, V10_a, V01_a = _delong_components(y, a)
    auc_b, V10_b, V01_b = _delong_components(y, b)
    m = len(V10_a)
    n = len(V01_a)

    # Variance and covariance of the (auc_a, auc_b) pair under DeLong.
    var_a = V10_a.var(ddof=1) / m + V01_a.var(ddof=1) / n
    var_b = V10_b.var(ddof=1) / m + V01_b.var(ddof=1) / n
    cov   = (np.cov(V10_a, V10_b, ddof=1)[0, 1] / m
             + np.cov(V01_a, V01_b, ddof=1)[0, 1] / n)
    var_diff = var_a + var_b - 2 * cov
    se = float(np.sqrt(max(var_diff, 0.0)))

    delta = float(auc_a - auc_b)
    if se == 0:
        p = 1.0 if delta == 0 else 0.0
        return float(auc_a), float(auc_b), delta, p, delta, delta

    z = abs(delta) / se
    p_value = 2 * (1 - float(stats.norm.cdf(z)))
    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    lo = delta - zcrit * se
    hi = delta + zcrit * se
    return float(auc_a), float(auc_b), delta, p_value, lo, hi


if __name__ == "__main__":
    # Smoke check: midrank-based AUC matches sklearn's, and the CI brackets it.
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(42)
    n = 500
    y = rng.binomial(1, 0.1, size=n)
    p1 = rng.beta(2, 5, size=n) + 0.4 * y
    p2 = rng.beta(2, 5, size=n) + 0.2 * y
    a1, lo1, hi1 = delong_auc_ci(y, p1)
    a2, lo2, hi2 = delong_auc_ci(y, p2)
    sk1 = roc_auc_score(y, p1)
    sk2 = roc_auc_score(y, p2)
    print(f"AUC1 (delong)  = {a1:.4f}   AUC1 (sklearn) = {sk1:.4f}   "
          f"95% CI = [{lo1:.4f}, {hi1:.4f}]")
    print(f"AUC2 (delong)  = {a2:.4f}   AUC2 (sklearn) = {sk2:.4f}   "
          f"95% CI = [{lo2:.4f}, {hi2:.4f}]")
    a, b, d, p, dl, dh = delong_paired_test(y, p1, p2)
    print(f"paired delta = {d:+.4f}   p = {p:.4g}   95% CI = [{dl:+.4f}, {dh:+.4f}]")
