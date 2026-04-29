import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "train_engineered.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val_engineered.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "stage0_model.joblib")


def load_and_prep_data(filepath):
    """Loads Parquet file and splits into features and target."""
    df = pd.read_parquet(filepath)
    # Drop identifier columns and the target
    cols_to_drop = ['entry_id', 'patient_id', 'date', 'migraine_target']
    X = df.drop(columns=cols_to_drop)
    y = df['migraine_target']
    return X, y


def main():
    print("Loading data...")
    X_train, y_train = load_and_prep_data(TRAIN_PATH)
    X_val, y_val = load_and_prep_data(VAL_PATH)

    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val set:   X={X_val.shape}, y={y_val.shape}")

    # 1. Define Base Estimators (XGBoost variants to provide diversity)
    base_estimators = [
        ('xgb_shallow',
         XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss')),
        ('xgb_deep',
         XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='logloss'))
    ]

    # 2. Define Meta-Model (L1-LR)
    meta_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=500)

    # 3. Build Stacking Ensemble
    print("Training Stacked Ensemble Baseline (XGBoost + L1-LR)...")
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    # Fit the stacker on the training data
    stacker.fit(X_train, y_train)

    # 4. Isotonic Calibration on Validation Set
    print("Applying Isotonic Calibration using Validation set...")
    calibrated_model = CalibratedClassifierCV(
        estimator=stacker,
        method='isotonic',
        cv='prefit'  # Prefit implies we only calibrate on the data passed to fit() below
    )

    # Fit calibration ONLY on validation data to prevent data leakage
    calibrated_model.fit(X_val, y_val)

    # 5. Save the final pipeline
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    joblib.dump(calibrated_model, MODEL_PATH)
    print(f"Model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()