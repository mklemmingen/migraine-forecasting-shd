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
    """Loads Parquet file and splits into features and target."""
    df = pd.read_parquet(filepath)
    X = df.drop(columns=['entry_id', 'patient_id', 'date', 'migraine_target'])
    y = df['migraine_target']
    return X, y


def main():
    print("Loading data...")
    X_train, y_train = load_and_prep_data(TRAIN_PATH)
    X_val, y_val = load_and_prep_data(VAL_PATH)

    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val set:   X={X_val.shape}, y={y_val.shape}")

    # 1. Initialize Foundation Model (TabPFN)
    # The dataset (n=3941, cols=40) fits perfectly within TabPFN's scaling limits.
    print("Initializing TabPFN Foundation Model...")
    tabpfn_base = TabPFNClassifier()

    # 2. "Fit" the model (In-context learning mapping)
    print("Fitting TabPFN on training set...")
    tabpfn_base.fit(X_train, y_train)

    # 3. Isotonic Calibration on Validation Set
    # Maintains the exact benchmark protocol established in Stage 0
    print("Applying Isotonic Calibration using Validation set...")
    calibrated_model = CalibratedClassifierCV(
        estimator=tabpfn_base,
        method='isotonic',
        cv='prefit'
    )

    # Fit calibration ONLY on validation data
    calibrated_model.fit(X_val, y_val)

    # 4. Save the pipeline
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    print(f"Model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()