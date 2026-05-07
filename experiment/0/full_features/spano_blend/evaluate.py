import os
import uuid
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, "..", "..", "..", "..", "data", "processed")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
VAL_PATH   = os.path.join(DATA_DIR, "70_15_15", "chrono", "diary_val.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "70_15_15", "chrono", "diary_test.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

RESULT_PREFIX = "results"
TITLE         = "STAGE 0 / full_features / spano_blend: SPANO BLEND ARCHITECTURE (train.py)"

# ---------------------------------------------------------------------------
# VALIDITY WARNING
# The Spano blend architecture uses the validation set for four sequential
# optimisation steps in train.py: fitting per-model isotonic and Platt
# calibrators, alpha grid search, and final calibrator selection. This script
# then reuses the same validation set for operating-threshold selection.
#
# Because the isotonic calibrators are fitted ON the val set they effectively
# memorise it — val probabilities after calibration approach the empirical
# positive rate within each predicted-probability group on val. Thresholds
# derived from these memorised probabilities are poorly matched to the test
# distribution, and ECE10 computed by resampling val-calibrated scores will be
# artificially low.
#
# Consequence: Sensitivity (>=0.5) and MCC (Optimal) metrics on test are
# unreliable. AUROC and AUPRC are rank-based and unaffected by calibration, so
# they remain the most trustworthy outputs of this script.
#
# The methodologically sound baseline for this benchmark is the stacked
# ensemble in ../clean_stack/evaluate.py (train.py), which uses only Platt
# (two-parameter) calibration on val and keeps threshold selection as the sole
# second use.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Calibrator classes — must match train.py exactly for pickle to resolve
# ---------------------------------------------------------------------------

class IsoCalibrator:
    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds='clip')

    def fit(self, proba, y):
        self._iso.fit(proba, y)
        return self

    def transform(self, proba):
        return self._iso.predict(proba)


class PlattCalibrator:
    def __init__(self):
        self._lr = LogisticRegression(fit_intercept=True, solver='lbfgs', max_iter=1000)

    def fit(self, proba, y):
        logit = np.log(np.clip(proba, 1e-8, 1 - 1e-8) / (1 - np.clip(proba, 1e-8, 1 - 1e-8)))
        self._lr.fit(logit.reshape(-1, 1), y)
        return self

    def transform(self, proba):
        logit = np.log(np.clip(proba, 1e-8, 1 - 1e-8) / (1 - np.clip(proba, 1e-8, 1 - 1e-8)))
        return self._lr.predict_proba(logit.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_prep_data(filepath):
    df = pd.read_parquet(filepath)
    X = df.drop(columns=['entry_id', 'patient_id', 'date', 'migraine_target'])
    y = df['migraine_target']
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0., 1., n_bins + 1)
    binned = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        mask = (binned == i)
        if mask.sum() > 0:
            ece += np.abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.sum()
    return ece / len(y_true)


def find_operating_thresholds(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    mccs    = [matthews_corrcoef(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    recalls = [recall_score(y_true,      (y_prob >= t).astype(int)) for t in thresholds]
    opt_mcc_thresh = thresholds[np.argmax(mccs)]
    valid = [t for t, r in zip(thresholds, recalls) if r >= 0.50]
    sens_05_thresh = max(valid) if valid else 0.50
    return opt_mcc_thresh, sens_05_thresh


def run_bootstrap_evaluation(y_true, y_prob, opt_mcc_thresh, sens_05_thresh, n_iterations=1000, seed=42):
    np.random.seed(seed)
    y_arr = y_true.values
    metrics = defaultdict(list)
    for _ in range(n_iterations):
        idx = np.random.randint(0, len(y_arr), len(y_arr))
        y_t, y_p = y_arr[idx], y_prob[idx]
        if len(np.unique(y_t)) < 2:
            continue
        metrics['AUROC'].append(roc_auc_score(y_t, y_p))
        metrics['AUPRC'].append(average_precision_score(y_t, y_p))
        metrics['Brier Score'].append(brier_score_loss(y_t, y_p))
        metrics['ECE10'].append(expected_calibration_error(y_t, y_p))
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _calibrated_proba(bundle, X):
    """Spano parallel blend: XGB + LR → per-model calibrators → alpha blend → final calibrator."""
    p_x_raw = bundle['xgb'].predict_proba(X)[:, 1]
    p_l_raw = bundle['lr_pipe'].predict_proba(X)[:, 1]
    if bundle['which_base_cal'] == 'iso+iso':
        p_x = bundle['cal_x_iso'].transform(p_x_raw)
        p_l = bundle['cal_l_iso'].transform(p_l_raw)
    else:
        p_x = bundle['cal_x_pl'].transform(p_x_raw)
        p_l = bundle['cal_l_pl'].transform(p_l_raw)
    p_blend = bundle['alpha'] * p_x + (1 - bundle['alpha']) * p_l
    return bundle['final_calibrator'].transform(p_blend)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading datasets and model...")
    X_val,  y_val  = load_and_prep_data(VAL_PATH)
    X_test, y_test = load_and_prep_data(TEST_PATH)
    bundle = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    y_prob_val  = _calibrated_proba(bundle, X_val)
    y_prob_test = _calibrated_proba(bundle, X_test)

    print("Calculating optimal thresholds on Validation set...")
    opt_mcc_thresh, sens_05_thresh = find_operating_thresholds(y_val, y_prob_val)

    print("Running bootstrap evaluation on locked Test set (n=1000)...")
    results = run_bootstrap_evaluation(y_test, y_prob_test, opt_mcc_thresh, sens_05_thresh)

    output_lines = [
        "=" * 60,
        TITLE,
        "=" * 60,
        "Validation Set Derived Thresholds:",
        f" -> MCC-Optimal Threshold:          {opt_mcc_thresh:.3f}",
        f" -> Threshold for Sens >= 0.50:     {sens_05_thresh:.3f}",
        "-" * 60,
        f"{'Metric':<25} | Mean [95% CI]",
        "-" * 60,
    ]
    for metric, result_str in results.items():
        output_lines.append(f"{metric:<25} | {result_str}")
    output_lines.append("=" * 60)
    output_text = "\n".join(output_lines)

    print("\n" + output_text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{RESULT_PREFIX}_{timestamp}_{str(uuid.uuid4())}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        f.write(output_text)
    print(f"\nResults successfully saved to: {filepath}")


if __name__ == "__main__":
    main()
