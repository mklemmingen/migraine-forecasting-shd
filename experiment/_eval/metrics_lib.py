"""Shared metric helpers for all leaf evaluators.

Single source of truth for the calibration, threshold-selection, and
bootstrap routines used by every Addition's hold-out and CV evaluator.
Centralising them here keeps numerical behaviour identical across leaves;
any change to a metric is picked up by all leaves on the next run.

Public surface:
    expected_calibration_error(y_true, y_prob, n_bins=10) -> float
    calibration_slope(y_true, y_prob) -> float
    find_operating_thresholds(y_true, y_prob) -> tuple[float, float]
    run_bootstrap_evaluation(y_true, y_prob, opt_mcc_thresh, sens_05_thresh,
                             n_iterations=1000, seed=42) -> dict[str, str]
    score_fold(y_val, p_val, opt_thresh, sens_thresh) -> dict[str, float]
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width probability bins.

    The fraction-weighted mean absolute difference between bin-empirical
    positive rate and bin-mean predicted probability.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binned    = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        mask = (binned == i)
        if mask.sum() > 0:
            ece += np.abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.sum()
    return ece / len(y_true)


def calibration_slope(y_true, y_prob) -> float:
    """Slope of the logistic recalibration line on the held-out set.

    Defined as the regression coefficient when refitting a single-feature
    logistic regression of the binary outcome on the logit of the model's
    predicted probabilities. Slope = 1 is perfect calibration; slope < 1
    indicates over-confidence (predictions more extreme than reality - the
    classical overfitting fingerprint); slope > 1 indicates
    under-confidence. Required by TRIPOD+AI (Collins et al., BMJ 2024)
    alongside ECE/Brier for a complete picture of calibration.
    """
    p     = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    # Effectively unpenalised logistic regression; class_weight=None on
    # purpose so the slope reflects raw recalibration, not class re-balancing.
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=2000)
    try:
        lr.fit(logit, np.asarray(y_true, dtype=int))
    except (ValueError, np.linalg.LinAlgError):
        return float('nan')
    return float(lr.coef_[0, 0])


def find_operating_thresholds(y_true, y_prob) -> tuple[float, float]:
    """Pick two operating thresholds from a 99-point sweep on [0.01, 0.99]:
    the MCC-optimal threshold, and the highest threshold whose recall is
    still >= 0.50 (falls back to 0.50 if no such threshold exists).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    mccs    = [matthews_corrcoef(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    recalls = [recall_score(y_true,      (y_prob >= t).astype(int)) for t in thresholds]
    opt_mcc_thresh = thresholds[np.argmax(mccs)]
    valid = [t for t, r in zip(thresholds, recalls) if r >= 0.50]
    sens_05_thresh = max(valid) if valid else 0.50
    return opt_mcc_thresh, sens_05_thresh


def run_bootstrap_evaluation(
    y_true,
    y_prob,
    opt_mcc_thresh: float,
    sens_05_thresh: float,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict[str, str]:
    """Bootstrap the locked test set ``n_iterations`` times.

    Each iteration resamples row indices with replacement, drops samples
    that lost class diversity, and scores the eleven metrics in the
    sharedMetricPrinter hold-out contract. Returns one ``"mean [lo - hi]"``
    string per metric, where lo/hi are the 2.5 / 97.5 percentiles.
    """
    np.random.seed(seed)
    y_arr   = y_true.values
    metrics = defaultdict(list)
    for _ in range(n_iterations):
        idx     = np.random.randint(0, len(y_arr), len(y_arr))
        y_t, y_p = y_arr[idx], y_prob[idx]
        if len(np.unique(y_t)) < 2:
            continue
        metrics['AUROC'].append(roc_auc_score(y_t, y_p))
        metrics['AUPRC'].append(average_precision_score(y_t, y_p))
        metrics['Brier Score'].append(brier_score_loss(y_t, y_p))
        metrics['ECE10'].append(expected_calibration_error(y_t, y_p))
        slope = calibration_slope(y_t, y_p)
        if np.isfinite(slope):
            metrics['Calibration Slope'].append(slope)
        preds_mcc = (y_p >= opt_mcc_thresh).astype(int)
        metrics['MCC (Optimal)'].append(matthews_corrcoef(y_t, preds_mcc))
        metrics['Sensitivity (>=0.5)'].append(
            recall_score(y_t, (y_p >= sens_05_thresh).astype(int)))
        metrics['Accuracy'].append(accuracy_score(y_t, preds_mcc))
        metrics['Precision'].append(precision_score(y_t, preds_mcc, zero_division=0))
        metrics['Recall'].append(recall_score(y_t, preds_mcc, zero_division=0))
        metrics['F1'].append(f1_score(y_t, preds_mcc, zero_division=0))
    return {
        name: f"{np.mean(v):.3f} [{np.percentile(v, 2.5):.3f} - {np.percentile(v, 97.5):.3f}]"
        for name, v in metrics.items()
    }


def score_fold(
    y_val,
    p_val,
    opt_thresh: float,
    sens_thresh: float,
) -> dict[str, float]:
    """Score a single CV fold against the two operating thresholds.

    Returns one numeric value per metric in the sharedMetricPrinter CV
    contract (the MCC label uses "Cal-Optimal" because the threshold is
    selected on the calibration sub-split, not the evaluation fold).
    """
    preds_opt = (p_val >= opt_thresh).astype(int)
    return {
        'AUROC':               roc_auc_score(y_val, p_val),
        'AUPRC':               average_precision_score(y_val, p_val),
        'Brier Score':         brier_score_loss(y_val, p_val),
        'ECE10':               expected_calibration_error(y_val, p_val),
        'Calibration Slope':   calibration_slope(y_val, p_val),
        'MCC (Cal-Optimal)':   matthews_corrcoef(y_val, preds_opt),
        'Sensitivity (>=0.5)': recall_score(y_val, (p_val >= sens_thresh).astype(int)),
        'Accuracy':            accuracy_score(y_val, preds_opt),
        'Precision':           precision_score(y_val, preds_opt, zero_division=0),
        'Recall':              recall_score(y_val, preds_opt, zero_division=0),
        'F1':                  f1_score(y_val, preds_opt, zero_division=0),
    }
