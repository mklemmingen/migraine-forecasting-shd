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
    cols_to_drop = ['entry_id', 'patient_id', 'date', 'migraine_target']
    X = df.drop(columns=cols_to_drop)
    y = df['migraine_target']
    return X, y


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

    # Imbalance ratio: weights the minority class (migraine ~5%) in both base learners
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Two XGBoost variants as base learners; architectural diversity (depth)
    # improves the meta-model's signal beyond what a single tree depth provides.
    base_estimators = [
        ('xgb_shallow', XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42)),
        ('xgb_deep', XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_state=42)),
    ]

    # L1-regularised meta-model; class_weight='balanced' compensates for the ~5% positive rate
    meta_model = LogisticRegression(
        l1_ratio=1.0, solver='saga', C=1.0,
        class_weight='balanced', random_state=42, max_iter=500)

    # KFold(shuffle=False) on chronologically sorted data produces 5 consecutive equal-sized
    # blocks — the closest partition-based approximation to time-series CV compatible with
    # StackingClassifier.  TimeSeriesSplit is incompatible because its expanding-window design
    # leaves early samples out of all test folds, violating the partition requirement of
    # cross_val_predict.  StratifiedKFold would be worse: random fold assignment maximises
    # temporal leakage.
    print("Training Stacked Ensemble Baseline (XGBoost + L1-LR)...")
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=KFold(n_splits=5, shuffle=False),
        n_jobs=-1,
    )
    stacker.fit(X_train, y_train)

    # Calibrate on held-out validation data so calibration error reflects
    # out-of-sample performance, not in-sample fit.
    print("Applying Sigmoid (Platt) Calibration using Validation set...")
    calibrator = fit_sigmoid_calibrator(stacker, X_val, y_val)

    joblib.dump({'stacker': stacker, 'calibrator': calibrator}, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
