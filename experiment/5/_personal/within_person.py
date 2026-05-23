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


def within_person_cstatistic(scores: pd.DataFrame) -> dict:
    """Precision-weighted random-effects within-person C-statistic.

    Combines the estimable per-patient AUROCs into a cohort summary, weighting
    short/noisy patient series less than long ones - the partial-pooling
    discipline Addition 3 adopted for clustered longitudinal estimates
    (docs/addition3_temporal.md Section 9), applied to discrimination. Per
    docs Section 9, Decision 1: per-patient variance via Hanley-McNeil,
    between-patient heterogeneity tau^2 via DerSimonian-Laird, and the
    random-effects weighted mean (the version to report, because the
    between-patient variation is the whole point, RQ3).

    Returns: estimate, ci_low, ci_high, tau2, k_estimable, median_auroc.
    """
    est = scores[scores["estimable"]].dropna(subset=["auroc"])
    k = len(est)
    base = {"k_estimable": int(k),
            "median_auroc": float(est["auroc"].median()) if k else float("nan")}
    if k == 0:
        return {**base, "estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "tau2": float("nan")}

    a = est["auroc"].to_numpy(dtype=float)
    n_pos = est["n_pos"].to_numpy(dtype=int)
    n_neg = (est["n"] - est["n_pos"]).to_numpy(dtype=int)
    v = np.array([_hanley_mcneil_var(ai, p, q) for ai, p, q in zip(a, n_pos, n_neg)])
    v = np.clip(v, 1e-6, None)

    w = 1.0 / v
    a_fe = float(np.sum(w * a) / np.sum(w))
    if k > 1:
        Q = float(np.sum(w * (a - a_fe) ** 2))
        C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    else:
        tau2 = 0.0
    ws = 1.0 / (v + tau2)
    est_re = float(np.sum(ws * a) / np.sum(ws))
    se_re = float(np.sqrt(1.0 / np.sum(ws)))
    return {**base, "estimate": est_re, "ci_low": est_re - 1.96 * se_re,
            "ci_high": est_re + 1.96 * se_re, "tau2": tau2}


def pooled_vs_within(pooled_auroc: float, within: dict) -> dict:
    """RQ2: the gap between the pooled headline AUROC (from comparison_*.html)
    and the within-person C-statistic. A large positive gap means the pooled
    number overstates per-patient forecasting skill [holsteen2020triggers,
    p. 2364]."""
    return {"pooled_auroc": pooled_auroc, "within_person": within,
            "gap": pooled_auroc - within.get("estimate", float("nan"))}
