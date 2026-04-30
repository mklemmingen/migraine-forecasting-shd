import os
import joblib
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.calibration import CalibratedClassifierCV

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "train_engineered.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val_engineered.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "stage1_model.joblib")


def load_and_prep_data(filepath):
    """Load a Parquet split and return (X, y) with identifier columns stripped."""
    df = pd.read_parquet(filepath)
    return prep_split(df)


def prep_split(df):
    """Return (X, y) from an already-loaded engineered DataFrame.

    Drops all non-feature columns so the same logic works for both the
    70/15/15 parquet files and the cv_engineered.parquet (which adds cv_fold).
    """
    drop_cols = ['entry_id', 'patient_id', 'date', 'migraine_target', 'cv_fold']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['migraine_target']
    return X, y


def build_tabpfn(X_train, y_train, X_cal, y_cal):
    """Fit TabPFN on training data and Platt-calibrate on a separate calibration set.

    Extracted from main() so evaluate_cv.py can re-train per fold without
    duplicating model configuration.

    Calibration uses a held-out cal set (not the evaluation fold) so that
    the evaluation fold is completely unseen at fit time.
    """
    tabpfn_base = TabPFNClassifier(device='cuda')
    tabpfn_base.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(
        estimator=tabpfn_base,
        method='sigmoid',
        cv='prefit',
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def main():
    print("Loading data...")
    X_train, y_train = load_and_prep_data(TRAIN_PATH)
    X_val, y_val = load_and_prep_data(VAL_PATH)

    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val set:   X={X_val.shape}, y={y_val.shape}")

    print("Fitting TabPFN + Platt calibration...")
    calibrated_model = build_tabpfn(X_train, y_train, X_val, y_val)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    print(f"Model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()