import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

# Shared imports
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import load_and_prep_data, prep_split, chronological_subsplit  # noqa: E402
from _dataRead.filter_to_no_rolling_features import remove_rolling_features  # noqa: E402
from _model_architecture.stacked_2xgb_meta_lr.model import calibrated_proba  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "migraine")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
TRAIN_PATH = os.path.join(DATA_DIR, "70_30", "stratified", "diary_train.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "70_30", "stratified", "diary_test.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

RESULT_PREFIX = "results"
TITLE         = "STAGE 0 / no_rolling_features / stacked_2xgb_meta_lr"

# 2-way ratio: no val parquet; reproduce the same chronological subsplit
# of train used during fitting to derive operating thresholds.
CAL_RATIO = 0.20


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


def main():
    print("Loading datasets and model...")
    df_train_full = remove_rolling_features(TRAIN_PATH)
    _, cal_sub = chronological_subsplit(df_train_full, cal_ratio=CAL_RATIO)
    X_cal, y_cal = prep_split(cal_sub)
    X_test, y_test = load_and_prep_data(TEST_PATH, loader=remove_rolling_features)
    bundle = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    y_prob_cal  = calibrated_proba(bundle, X_cal)
    y_prob_test = calibrated_proba(bundle, X_test)

    print("Calculating optimal thresholds on cal sub-split...")
    opt_mcc_thresh, sens_05_thresh = find_operating_thresholds(y_cal, y_prob_cal)

    print("Running bootstrap evaluation on locked Test set (n=1000)...")
    results = run_bootstrap_evaluation(y_test, y_prob_test, opt_mcc_thresh, sens_05_thresh)

    output_lines = [
        "=" * 60,
        TITLE,
        "=" * 60,
        "Cal Sub-Split Derived Thresholds:",
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
