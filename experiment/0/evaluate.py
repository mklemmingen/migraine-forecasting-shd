import os
import joblib
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    recall_score
)

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
VAL_PATH = os.path.join(DATA_DIR, "val_engineered.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test_engineered.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "stage0_model.joblib")


def load_and_prep_data(filepath):
    df = pd.read_parquet(filepath)
    X = df.drop(columns=['entry_id', 'patient_id', 'date', 'migraine_target'])
    y = df['migraine_target']
    return X, y


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculates ECE10 (Expected Calibration Error with 10 bins)."""
    bin_edges = np.linspace(0., 1., n_bins + 1)
    binned = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    for i in range(n_bins):
        bin_mask = (binned == i)
        if bin_mask.sum() > 0:
            acc = y_true[bin_mask].mean()
            conf = y_prob[bin_mask].mean()
            ece += np.abs(acc - conf) * bin_mask.sum()

    return ece / len(y_true)


def find_operating_thresholds(y_true, y_prob):
    """Finds MCC-optimal and Sensitivity >= 0.50 thresholds."""
    thresholds = np.linspace(0.01, 0.99, 99)
    mccs = []
    recalls = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        mccs.append(matthews_corrcoef(y_true, preds))
        recalls.append(recall_score(y_true, preds))

    opt_mcc_thresh = thresholds[np.argmax(mccs)]

    # Find highest threshold that maintains sensitivity >= 0.50
    valid_sens_thresholds = [t for t, r in zip(thresholds, recalls) if r >= 0.50]
    sens_05_thresh = max(valid_sens_thresholds) if valid_sens_thresholds else 0.50

    return opt_mcc_thresh, sens_05_thresh


def run_bootstrap_evaluation(y_true, y_prob, opt_mcc_thresh, sens_05_thresh, n_iterations=1000, seed=42):
    """Runs bootstrap resampling to calculate 95% Confidence Intervals."""
    np.random.seed(seed)
    n_size = len(y_true)
    y_true_arr = y_true.values

    metrics = defaultdict(list)

    for _ in range(n_iterations):
        idx = np.random.randint(0, n_size, n_size)
        y_t, y_p = y_true_arr[idx], y_prob[idx]

        # Skip pathological samples where only one class is present
        if len(np.unique(y_t)) < 2:
            continue

        metrics['AUROC'].append(roc_auc_score(y_t, y_p))
        metrics['AUPRC'].append(average_precision_score(y_t, y_p))
        metrics['Brier Score'].append(brier_score_loss(y_t, y_p))
        metrics['ECE10'].append(expected_calibration_error(y_t, y_p, n_bins=10))

        # Operating points
        preds_mcc = (y_p >= opt_mcc_thresh).astype(int)
        metrics['MCC (Optimal)'].append(matthews_corrcoef(y_t, preds_mcc))

        preds_sens = (y_p >= sens_05_thresh).astype(int)
        metrics['Sensitivity (>=0.5)'].append(recall_score(y_t, preds_sens))

    # Aggregate results
    results = {}
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        lower_ci = np.percentile(values, 2.5)
        upper_ci = np.percentile(values, 97.5)
        results[metric_name] = f"{mean_val:.3f} [{lower_ci:.3f} - {upper_ci:.3f}]"

    return results


def main():
    print("Loading datasets and model...")
    X_val, y_val = load_and_prep_data(VAL_PATH)
    X_test, y_test = load_and_prep_data(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    # Predict probabilities (extracting probability of class 1)
    print("Generating predictions...")
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Find thresholds strictly on Validation set
    print("Calculating optimal thresholds on Validation set...")
    opt_mcc_thresh, sens_05_thresh = find_operating_thresholds(y_val, y_prob_val)

    # Run Bootstrapping on Test Set
    print("Running bootstrap evaluation on locked Test set (n=1000)...")
    results = run_bootstrap_evaluation(y_test, y_prob_test, opt_mcc_thresh, sens_05_thresh)

    # Build the formatted output string
    output_lines = [
        "=" * 60,
        "STAGE 0: STACKED ENSEMBLE BASELINE RESULTS",
        "=" * 60,
        f"Validation Set Derived Thresholds:",
        f" -> MCC-Optimal Threshold:          {opt_mcc_thresh:.3f}",
        f" -> Threshold for Sens >= 0.50:     {sens_05_thresh:.3f}",
        "-" * 60,
        f"{'Metric':<25} | Mean [95% CI]",
        "-" * 60
    ]

    for metric, result_str in results.items():
        output_lines.append(f"{metric:<25} | {result_str}")

    output_lines.append("=" * 60)
    output_text = "\n".join(output_lines)

    # Print output to console
    print("\n" + output_text)

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate timestamp with milliseconds (YYYYMMDD_HHMMSS_mmm) and UUID
    # %f gives microseconds, so we slice [:-3] to get exactly 3 digits for milliseconds
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_uuid = str(uuid.uuid4())

    filename = f"results_{timestamp}_{run_uuid}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)

    # Write to file
    with open(filepath, "w") as f:
        f.write(output_text)

    print(f"\nResults successfully saved to: {filepath}")


if __name__ == "__main__":
    main()