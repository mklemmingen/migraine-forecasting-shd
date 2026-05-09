import os
import sys

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Architecture import
_EXP0 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', '..', '..', '..', '..'))
sys.path.insert(0, _EXP0)
from _model_architecture.clean_stack.model import (  # noqa: E402
    build_stacker, fit_sigmoid_calibrator,
)

# ---------------------------------------------------------------------------
# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXPERIMENT_DIR, "..", "..", "..", "..", "..", "..", "..", "data", "processed", "headache")
TRAIN_PATH = os.path.join(DATA_DIR, "70_15_15", "chrono", "diary_train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "70_15_15", "chrono", "diary_val.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")
# ---------------------------------------------------------------------------

def load_and_prep_data(filepath):
    """Load a Parquet split and return (X, y) with identifier columns stripped."""
    df = pd.read_parquet(filepath)
    return prep_split(df)


def prep_split(df):
    """Return (X, y) from the loaded parquet -> data frame

    Drops all non-feature columns so the same logic works for both the
    70/15/15 parquet files and the cv_engineered.parquet (which adds cv_fold).
    """
    drop_cols = ['entry_id', 'patient_id', 'date', 'migraine_target', 'cv_fold']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['migraine_target']
    return X, y


def main():
    print("Loading data...")
    X_train, y_train = load_and_prep_data(TRAIN_PATH)
    X_val, y_val = load_and_prep_data(VAL_PATH)

    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val set:   X={X_val.shape}, y={y_val.shape}")

    print("Training Stacked Ensemble Baseline (XGBoost + L1-LR)...")
    stacker = build_stacker(X_train, y_train)

    print("Applying Sigmoid (Platt) Calibration using Validation set...")
    calibrator = fit_sigmoid_calibrator(stacker, X_val, y_val)

    joblib.dump({'stacker': stacker, 'calibrator': calibrator}, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
