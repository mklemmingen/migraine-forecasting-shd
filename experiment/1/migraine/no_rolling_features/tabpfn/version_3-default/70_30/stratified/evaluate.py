import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import joblib

# Shared imports
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import load_and_prep_data, prep_split, chronological_subsplit  # noqa: E402
from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402

from _eval.metrics_lib import find_operating_thresholds, run_bootstrap_evaluation  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "migraine")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
TRAIN_PATH = os.path.join(DATA_DIR, "70_30", "stratified", "diary_train.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "70_30", "stratified", "diary_test.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

RESULT_PREFIX = "results"
TITLE         = "STAGE 1 / no_rolling_features / tabpfn (version_3-default)"

# 2-way ratio: no val parquet; reproduce the same chronological subsplit
# of train used during fitting to derive operating thresholds.
CAL_RATIO = 0.20


def main():
    print("Loading datasets and model...")
    df_train_full = select_non_rolling_features(TRAIN_PATH)
    _, cal_sub = chronological_subsplit(df_train_full, cal_ratio=CAL_RATIO)
    X_cal, y_cal = prep_split(cal_sub)
    X_test, y_test = load_and_prep_data(TEST_PATH, loader=select_non_rolling_features)
    model = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    y_prob_cal  = model.predict_proba(X_cal)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

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
