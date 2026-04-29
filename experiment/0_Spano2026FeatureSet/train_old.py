"""
Replication of Marco Samuel Spanos Bachelor-Thesis Architecture as stated in the Conference Paper (Spano 2026) on the benchmark dataset.

Architecture: two independent base models (XGBoost + L1-LR pipeline) blended by a
val-MCC alpha grid search, each with per-model isotonic and Platt calibrators, followed
by a final calibrator also selected by val MCC.

Kept separate from train.py to allow direct metric comparison.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Cleaner of Trigger columns that were not in Spano 2026, to ensure a fair comparison with the original architecture.
# See parquetFilterToOldFeatureSet.py for details.
from parquetFilterToOldFeatureSet import remove_non_spano_features

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "train_engineered.parquet")
VAL_PATH   = os.path.join(DATA_DIR, "val_engineered.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "../0_Spano2026FeatureSet/stage0_model_old.joblib")

ALPHA_GRID = np.linspace(0.0, 1.0, 21)   # 0_FullSHD18TriggerFeatureSet.00, 0_FullSHD18TriggerFeatureSet.05, ..., 1.00  (Spano config.py)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_prep_data(filepath):
    """Load a Parquet split and return (X, y) with identifier columns stripped."""
    df = remove_non_spano_features(filepath)
    cols_to_drop = ['entry_id', 'patient_id', 'date', 'migraine_target']
    X = df.drop(columns=cols_to_drop)
    y = df['migraine_target']
    return X, y


# ---------------------------------------------------------------------------
# Calibrators  (faithful to calibration.py in final_model)
# ---------------------------------------------------------------------------

class IsoCalibrator:
    """Non-parametric monotone calibration via isotonic regression."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds='clip')

    def fit(self, proba, y):
        self._iso.fit(proba, y)
        return self

    def transform(self, proba):
        return self._iso.predict(proba)


class PlattCalibrator:
    """Platt scaling in logit space: fits LR on log(p/(1-p)) (Platt 1999)."""

    def __init__(self):
        self._lr = LogisticRegression(fit_intercept=True, solver='lbfgs', max_iter=1000)

    def fit(self, proba, y):
        logit = np.log(np.clip(proba, 1e-8, 1 - 1e-8) / (1 - np.clip(proba, 1e-8, 1 - 1e-8)))
        self._lr.fit(logit.reshape(-1, 1), y)
        return self

    def transform(self, proba):
        logit = np.log(np.clip(proba, 1e-8, 1 - 1e-8) / (1 - np.clip(proba, 1e-8, 1 - 1e-8)))
        return self._lr.predict_proba(logit.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Base models  (hyperparameters from models.py in final_model)
# ---------------------------------------------------------------------------

def fit_xgb(X_tr, y_tr, scale_pos_weight):
    """XGBoost with Spano hyperparameters and class-imbalance weight."""
    model = XGBClassifier(
        n_estimators=800,        # Spano fallback (early-stopping target was 2000)
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.6,
        reg_lambda=2.0,
        objective='binary:logistic',
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    return model


def fit_lr_pipeline(X_tr, y_tr, class_weight):
    """L1-LR inside a preprocessing pipeline (imputer + scaler), as in Spano."""
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('lr',      LogisticRegression(
            l1_ratio=1.0, solver='saga', C=1.0,
            class_weight=class_weight,
            random_state=42, max_iter=5000,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


# ---------------------------------------------------------------------------
# Blending and final calibration  (logic from train.py in final_model)
# ---------------------------------------------------------------------------

def _best_mcc_over_thresholds(p, y):
    """MCC at the oracle threshold — used only for model selection on val."""
    return max(matthews_corrcoef(y, (p >= t).astype(int)) for t in np.unique(p))


def search_alpha(p_x_cal, p_l_cal, y):
    """Return the alpha in ALPHA_GRID that maximises val MCC."""
    best_alpha, best_mcc = 0.5, -1.0
    for alpha in ALPHA_GRID:
        p_blend = alpha * p_x_cal + (1 - alpha) * p_l_cal
        mcc = _best_mcc_over_thresholds(p_blend, y)
        if mcc > best_mcc:
            best_mcc, best_alpha = mcc, alpha
    return best_alpha, best_mcc


def select_final_calibrator(p_blend_val, y_val):
    """Fit both calibrators on blended val probs; return the one with higher oracle MCC."""
    iso = IsoCalibrator().fit(p_blend_val, y_val)
    plt = PlattCalibrator().fit(p_blend_val, y_val)
    mcc_iso = _best_mcc_over_thresholds(iso.transform(p_blend_val), y_val)
    mcc_plt = _best_mcc_over_thresholds(plt.transform(p_blend_val), y_val)
    if mcc_iso >= mcc_plt:
        return iso, 'isotonic'
    return plt, 'platt'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    X_train, y_train = load_and_prep_data(TRAIN_PATH)
    X_val,   y_val   = load_and_prep_data(VAL_PATH)

    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val set:   X={X_val.shape}, y={y_val.shape}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = float(neg / pos)
    class_weight = {0: 1.0, 1: scale_pos_weight}

    print("Fitting XGBoost base model...")
    xgb = fit_xgb(X_train, y_train, scale_pos_weight)

    print("Fitting L1-LR pipeline...")
    lr_pipe = fit_lr_pipeline(X_train, y_train, class_weight)

    # Per-model calibration on validation set
    print("Calibrating base models on validation set...")
    p_x_raw = xgb.predict_proba(X_val)[:, 1]
    p_l_raw = lr_pipe.predict_proba(X_val)[:, 1]

    cal_x_iso = IsoCalibrator().fit(p_x_raw, y_val.values)
    cal_x_pl  = PlattCalibrator().fit(p_x_raw, y_val.values)
    cal_l_iso = IsoCalibrator().fit(p_l_raw, y_val.values)
    cal_l_pl  = PlattCalibrator().fit(p_l_raw, y_val.values)

    # Alpha grid search over both calibrator combos, select by val MCC
    print("Searching alpha grid over iso+iso and sig+sig combos...")
    combos = {
        'iso+iso': (cal_x_iso.transform(p_x_raw), cal_l_iso.transform(p_l_raw)),
        'sig+sig': (cal_x_pl.transform(p_x_raw),  cal_l_pl.transform(p_l_raw)),
    }

    best_combo, best_alpha, best_mcc = None, 0.5, -1.0
    for name, (p_x_cal, p_l_cal) in combos.items():
        alpha, mcc = search_alpha(p_x_cal, p_l_cal, y_val.values)
        if mcc > best_mcc:
            best_mcc, best_combo, best_alpha = mcc, name, alpha

    print(f"Selected: combo={best_combo}, alpha={best_alpha:.2f}, oracle val MCC={best_mcc:.3f}")

    p_x_cal   = combos[best_combo][0]
    p_l_cal   = combos[best_combo][1]
    p_blend   = best_alpha * p_x_cal + (1 - best_alpha) * p_l_cal

    print("Fitting final calibrator...")
    final_cal, final_cal_name = select_final_calibrator(p_blend, y_val.values)
    print(f"Final calibrator: {final_cal_name}")

    joblib.dump({
        'bundle_type':          'spano_blend',
        'xgb':                  xgb,
        'lr_pipe':              lr_pipe,
        'cal_x_iso':            cal_x_iso,
        'cal_x_pl':             cal_x_pl,
        'cal_l_iso':            cal_l_iso,
        'cal_l_pl':             cal_l_pl,
        'alpha':                best_alpha,
        'which_base_cal':       best_combo,
        'final_calibrator':     final_cal,
        'final_calibrator_name': final_cal_name,
    }, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
