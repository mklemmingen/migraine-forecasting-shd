import os

import joblib
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "train_engineered.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val_engineered.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "stage0_model.joblib")
# ---------------------------------------------------------------------------

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


def build_stacker(X_train, y_train):
    """Construct and fit the stacked ensemble on the supplied training data.

    Extracted from main() so evaluate_cv.py can re-train per fold without
    duplicating model configuration.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    base_estimators = [
        ('xgb_shallow', XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42)),
        ('xgb_deep', XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42)),
    ]

    meta_model = LogisticRegression(
        l1_ratio=1.0, solver='saga', C=1.0,
        class_weight='balanced', random_state=42, max_iter=500)

    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=KFold(n_splits=5, shuffle=False),
        n_jobs=-1,
    )
    stacker.fit(X_train, y_train)
    return stacker


def fit_sigmoid_calibrator(stacker, X_cal, y_cal):
    """Fit a Platt scaling calibrator on held-out probabilities from a pre-fitted stacker.

    Maps raw stacker probabilities via logistic regression so that
    calibrated_prob(x) ≈ P(y=1 | x).  Using a separate calibration set
    prevents training-data leakage into the calibration step.

    Sigmoid (Platt scaling) is preferred over isotonic regression when
    calibration samples are scarce.  Isotonic regression requires ≥1000 samples
    (Caruana et al. 2005); with val n=439 and ~5% positive rate (~23 positives),
    a two-parameter sigmoid fit is the statistically appropriate choice.
    """
    probs = stacker.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    # C=1e10 ≈ no regularisation — standard Platt scaling parameterisation
    calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrator.fit(probs, y_cal)
    return calibrator


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
