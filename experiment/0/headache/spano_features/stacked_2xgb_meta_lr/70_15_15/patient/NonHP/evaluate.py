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
from _dataRead.read import load_and_prep_data as _load_and_prep_data, prep_split  # noqa: E402
from _dataRead.filter_to_spano_features import select_spano_features  # noqa: E402
from _model_architecture.stacked_2xgb_meta_lr.model import calibrated_proba  # noqa: E402
from _eval.metrics_lib import find_operating_thresholds, run_bootstrap_evaluation  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "headache")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
VAL_PATH   = os.path.join(DATA_DIR, "70_15_15", "patient", "diary_val.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "70_15_15", "patient", "diary_test.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

RESULT_PREFIX = "results"
TITLE         = "STAGE 0 / spano_features / stacked_2xgb_meta_lr"
ARCH_FAMILY   = "xgboost"
# Off-by-default explainability flag. A normal sweep never sets it, so the
# attribution / effect cost is paid only on the selected insight leaves.
EMIT_INSIGHTS = os.environ.get("EMIT_INSIGHTS", "0") == "1"


def load_and_prep_data(filepath):
    """Spano-feature variant: whitelist Spano (2026) columns before (X, y) split."""
    return _load_and_prep_data(filepath, loader=select_spano_features)



def main():
    print("Loading datasets and model...")
    X_val,  y_val  = load_and_prep_data(VAL_PATH)
    X_test, y_test = load_and_prep_data(TEST_PATH)
    bundle = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    y_prob_val  = calibrated_proba(bundle, X_val)
    y_prob_test = calibrated_proba(bundle, X_test)

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

    if EMIT_INSIGHTS:
        # SHAP background is the val set already in scope (never the train
        # set). The closure explains the calibrated probability the
        # benchmark scores; the bundle is passed as raw_model for parity
        # with the TabPFN paths (unused by the XGBoost dispatch).
        from _explain import emit_insights
        emit_insights(
            predict_fn=lambda X: calibrated_proba(bundle, X),
            X_background=X_val,
            X_explain=X_test,
            y_explain=y_test,
            leaf_dir=EXPERIMENT_DIR,
            arch_family=ARCH_FAMILY,
            raw_model=bundle,
            title=TITLE,
            y_background=y_val,
        )


if __name__ == "__main__":
    main()
