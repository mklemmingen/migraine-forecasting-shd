import os
import uuid
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, "..", "..", "..", "..", "data")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
VAL_PATH   = os.path.join(DATA_DIR, "val_engineered.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "test_engineered.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

RESULT_PREFIX = "results"
TITLE         = "STAGE 0 / full_features / clean_stack: STACKED ENSEMBLE (train.py)"


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
        metrics['MCC (Optimal)'].append(
            matthews_corrcoef(y_t, (y_p >= opt_mcc_thresh).astype(int)))
        metrics['Sensitivity (>=0.5)'].append(
            recall_score(y_t, (y_p >= sens_05_thresh).astype(int)))
    return {
        name: f"{np.mean(v):.3f} [{np.percentile(v, 2.5):.3f} - {np.percentile(v, 97.5):.3f}]"
        for name, v in metrics.items()
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _calibrated_proba(bundle, X):
    """Stacked ensemble + Platt (or isotonic) calibrator saved by train.py."""
    raw = bundle['stacker'].predict_proba(X)[:, 1]
    cal = bundle['calibrator']
    if hasattr(cal, 'predict_proba'):
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    return cal.predict(raw)


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
