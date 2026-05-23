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


def within_person_cstatistic(scores: pd.DataFrame) -> dict:
    """Precision-weighted within-person C-statistic across patients.

    Combines the estimable per-patient AUROCs into a cohort summary, weighting
    short/noisy patient series less than long ones - the partial-pooling
    discipline Addition 3 adopted for clustered longitudinal estimates
    (docs/addition3_temporal.md Section 9), applied here to discrimination.

    TODO (decision: aggregation, docs Section 9):
        Implement the precision-weighted random-effects pool:
          - per-patient AUROC variance via the Hanley-McNeil approximation
            (function of n_pos, n_neg), giving weight w_i = 1 / (var_i + tau^2);
          - between-patient heterogeneity tau^2 by DerSimonian-Laird;
          - report the weighted mean and its 95% CI.
        The inverse-variance (fixed-effect) special case (tau^2 = 0) is the
        minimal version; the random-effects version is the one to report because
        between-patient variation is the whole point (RQ3). ~15-20 lines.
        Until implemented, callers can fall back to the unweighted median of the
        estimable per-patient AUROCs as a placeholder summary.
    """
    raise NotImplementedError("within-person C-statistic aggregation - see TODO")


def pooled_vs_within(pooled_auroc: float, within: dict) -> dict:
    """RQ2: the gap between the pooled headline AUROC (from comparison_*.html)
    and the within-person C-statistic. A large positive gap means the pooled
    number overstates per-patient forecasting skill [holsteen2020triggers,
    p. 2364]."""
    return {"pooled_auroc": pooled_auroc, "within_person": within,
            "gap": pooled_auroc - within.get("estimate", float("nan"))}
