"""Within-person evaluation - the core of Addition 5.

The pooled C-statistic mixes between-patient base-rate variation into the
discrimination estimate and overstates within-person forecasting skill; the
fitting evaluation for individual-attack forecasting is the within-person
metric, the meta-analytic summary of the person-specific discriminations
[holsteen2020triggers, p. 2364]. This module turns a (y_true, y_prob,
patient_id) triple - obtained by re-predicting an existing leaf's model on its
test split - into the per-patient AUROC/AUPRC distribution and a
precision-weighted within-person C-statistic.

Runs on the EXISTING Addition 0/1/4 predictions with no new training, so it
also yields the pooled-vs-within-person gap (RQ2) directly.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Minimum positive days for a per-patient AUROC to be estimable. Below this the
# per-patient estimate is too noisy to report (Addition 3 sparsity caveat;
# martin2025samplesize p. 2). Patients below the floor are flagged not-estimable
# rather than shrunk to a fabricated value. Decision in docs Section 9.
MIN_POS = 5


def per_patient_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_id: np.ndarray,
    min_pos: int = MIN_POS,
) -> pd.DataFrame:
    """Per-patient discrimination table.

    Returns one row per patient with columns: patient, n, n_pos, auroc, auprc,
    estimable. AUROC/AUPRC are NaN where the patient has < min_pos positives or
    is single-class (not estimable).
    """
    df = pd.DataFrame({"patient": patient_id, "y": y_true, "p": y_prob})
    rows = []
    for pid, g in df.groupby("patient", sort=True):
        n, n_pos = len(g), int(g["y"].sum())
        estimable = n_pos >= min_pos and 0 < n_pos < n
        auroc = roc_auc_score(g["y"], g["p"]) if estimable else np.nan
        auprc = average_precision_score(g["y"], g["p"]) if estimable else np.nan
        rows.append((pid, n, n_pos, auroc, auprc, estimable))
    return pd.DataFrame(rows, columns=["patient", "n", "n_pos", "auroc", "auprc", "estimable"])


def _hanley_mcneil_var(auc: float, n_pos: int, n_neg: int) -> float:
    """Variance of a single AUROC estimate (Hanley & McNeil 1982).

    var = [A(1-A) + (m-1)(Q1 - A^2) + (n-1)(Q2 - A^2)] / (m n), with
    Q1 = A/(2-A), Q2 = 2A^2/(1+A), m = n_pos, n = n_neg.
    """
    a = float(auc)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    num = a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)
    return num / (n_pos * n_neg)


def _pm_tau2(a: np.ndarray, v: np.ndarray,
             max_iter: int = 100, tol: float = 1e-8) -> float:
    """Paule-Mandel τ² estimator via bisection on the PM equation
    ``Σ w_i (a_i - μ_w)² = k - 1`` where ``w_i = 1/(v_i + τ²)`` and
    ``μ_w = Σ w_i a_i / Σ w_i``. The PM equation always has a non-negative
    solution; if Q(0) ≤ k-1 then τ²=0 is the optimum (no between-patient
    heterogeneity beyond Hanley-McNeil within-patient noise). Preferred over
    DerSimonian-Laird at k < 20 per Veroniki et al. 2016 (Res Synth Methods
    7:55-79) since DL systematically underestimates τ² at small k."""
    k = len(a)
    if k <= 1:
        return 0.0

    def q_gen(t2: float) -> float:
        w = 1.0 / (v + t2)
        mu = float(np.sum(w * a) / np.sum(w))
        return float(np.sum(w * (a - mu) ** 2))

    if q_gen(0.0) <= k - 1 + tol:
        return 0.0

    lo, hi = 0.0, max(float(np.var(a, ddof=1)) * 4.0, 1.0)
    # Ensure upper bound brackets the root (q is monotonically decreasing in τ²).
    while q_gen(hi) > k - 1:
        hi *= 2.0
        if hi > 1e6:
            return hi  # safety bound; pathological case

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        q = q_gen(mid)
        if abs(q - (k - 1)) < tol:
            return mid
        if q > k - 1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def within_person_cstatistic(scores: pd.DataFrame, method: str = "PM") -> dict:
    """Precision-weighted random-effects within-person C-statistic.

    Combines the estimable per-patient AUROCs into a cohort summary, weighting
    short/noisy patient series less than long ones - the partial-pooling
    discipline Addition 3 adopted for clustered longitudinal estimates
    (docs/addition3_temporal.md Section 9), applied to discrimination. Per-
    patient variance via Hanley-McNeil; between-patient heterogeneity τ²
    via either Paule-Mandel (default, preferred under k < 20 per body §2.4)
    or DerSimonian-Laird (legacy default, reported here as a sensitivity).

    Args:
        scores: per_patient_scores output (one row per patient).
        method: "PM" (Paule-Mandel, default) or "DL" (DerSimonian-Laird).

    Returns: estimate, ci_low, ci_high, tau2, tau2_method (the chosen
    method's primary results), plus tau2_dl, estimate_dl, ci_low_dl,
    ci_high_dl, tau2_pm, estimate_pm, ci_low_pm, ci_high_pm (always both
    methods' values for side-by-side reporting), plus k_estimable,
    median_auroc.
    """
    est = scores[scores["estimable"]].dropna(subset=["auroc"])
    k = len(est)
    nan = float("nan")
    base = {"k_estimable": int(k),
            "median_auroc": float(est["auroc"].median()) if k else nan}
    if k == 0:
        return {**base, "estimate": nan, "ci_low": nan, "ci_high": nan,
                "tau2": nan, "tau2_method": method.upper(),
                "tau2_dl": nan, "estimate_dl": nan,
                "ci_low_dl": nan, "ci_high_dl": nan,
                "tau2_pm": nan, "estimate_pm": nan,
                "ci_low_pm": nan, "ci_high_pm": nan}

    a = est["auroc"].to_numpy(dtype=float)
    n_pos = est["n_pos"].to_numpy(dtype=int)
    n_neg = (est["n"] - est["n_pos"]).to_numpy(dtype=int)
    v = np.array([_hanley_mcneil_var(ai, p, q) for ai, p, q in zip(a, n_pos, n_neg)])
    v = np.clip(v, 1e-6, None)

    # DerSimonian-Laird τ² (legacy sensitivity)
    w = 1.0 / v
    a_fe = float(np.sum(w * a) / np.sum(w))
    if k > 1:
        Q = float(np.sum(w * (a - a_fe) ** 2))
        C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2_dl = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    else:
        tau2_dl = 0.0
    w_dl = 1.0 / (v + tau2_dl)
    est_dl = float(np.sum(w_dl * a) / np.sum(w_dl))
    se_dl = float(np.sqrt(1.0 / np.sum(w_dl)))

    # Paule-Mandel τ² (primary)
    tau2_pm = _pm_tau2(a, v) if k > 1 else 0.0
    w_pm = 1.0 / (v + tau2_pm)
    est_pm = float(np.sum(w_pm * a) / np.sum(w_pm))
    se_pm = float(np.sqrt(1.0 / np.sum(w_pm)))

    method_u = method.upper()
    if method_u in ("PM", "PAULE_MANDEL", "PAULEMANDEL"):
        primary_est, primary_tau2, primary_se = est_pm, tau2_pm, se_pm
    elif method_u in ("DL", "DERSIMONIAN_LAIRD", "DERSIMONIANLAIRD"):
        primary_est, primary_tau2, primary_se = est_dl, tau2_dl, se_dl
    else:
        raise ValueError(f"unknown method: {method!r}; expected 'PM' or 'DL'")

    return {**base,
            "estimate": primary_est,
            "ci_low": primary_est - 1.96 * primary_se,
            "ci_high": primary_est + 1.96 * primary_se,
            "tau2": primary_tau2,
            "tau2_method": method_u,
            "tau2_dl": tau2_dl,
            "estimate_dl": est_dl,
            "ci_low_dl": est_dl - 1.96 * se_dl,
            "ci_high_dl": est_dl + 1.96 * se_dl,
            "tau2_pm": tau2_pm,
            "estimate_pm": est_pm,
            "ci_low_pm": est_pm - 1.96 * se_pm,
            "ci_high_pm": est_pm + 1.96 * se_pm}


def pooled_vs_within(pooled_auroc: float, within: dict) -> dict:
    """RQ2: the gap between the pooled headline AUROC (from comparison_*.html)
    and the within-person C-statistic. A large positive gap means the pooled
    number overstates per-patient forecasting skill [holsteen2020triggers,
    p. 2364]."""
    return {"pooled_auroc": pooled_auroc, "within_person": within,
            "gap": pooled_auroc - within.get("estimate", float("nan"))}
