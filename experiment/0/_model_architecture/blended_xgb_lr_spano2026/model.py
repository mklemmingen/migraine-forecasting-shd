"""
Blended XGBoost + L1-LR architecture (Spano 2026 replication).

Two independent base models - XGBoost and an L1-LR pipeline - are calibrated
per-base (isotonic + Platt) on a held-out split, then blended via convex
combination. The blend ratio (alpha) and the post-blend final calibrator are
both selected by best validation MCC.

Methodological caveat
---------------------
The same held-out split drives FOUR sequential optimisation steps:
    1. Per-base isotonic + Platt calibrators (4 fits)
    2. Alpha grid search (combinatorial over base-calibrator combos)
    3. Final-calibrator selection
    4. (Downstream) operating-threshold selection in evaluate.py
Because the isotonic calibrators effectively memorise the calibration set,
threshold-derived metrics (Sensitivity ≥ 0.5, MCC) on test are unreliable.
AUROC and AUPRC remain trustworthy (rank-based, calibration-invariant).

This architecture is preserved as a faithful replication of the prior
bachelor-thesis baseline; the methodologically clean comparator is
stacked_2xgb_meta_lr.

Public API
----------
build_model(X_train, y_train, X_val, y_val) -> bundle
calibrated_proba(bundle, X) -> ndarray

The bundle is a dict picklable via joblib. Calibrator classes
(IsoCalibrator, PlattCalibrator) live in this module so joblib.load
resolves them at the same import path on every leaf.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ALPHA_GRID = np.linspace(0.0, 1.0, 21)   # 0, 0.05, …, 1.00 (Spano config.py)


# ---------------------------------------------------------------------------
# Calibrators - class identities pickled into the bundle
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
    """Platt scaling in logit space: fits LR on log(p/(1-p))."""

    def __init__(self):
        self._lr = LogisticRegression(fit_intercept=True, solver='lbfgs', max_iter=1000)

    @staticmethod
    def _logit(proba):
        p = np.clip(proba, 1e-8, 1 - 1e-8)
        return np.log(p / (1 - p))

    def fit(self, proba, y):
        self._lr.fit(self._logit(proba).reshape(-1, 1), y)
        return self

    def transform(self, proba):
        return self._lr.predict_proba(self._logit(proba).reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Base models (Spano final_model/models.py hyperparameters)
# ---------------------------------------------------------------------------

def fit_xgb(X_tr, y_tr, scale_pos_weight):
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
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('lr',      LogisticRegression(
            penalty='l1', solver='saga', C=1.0,
            class_weight=class_weight,
            random_state=42, max_iter=5000,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


# ---------------------------------------------------------------------------
# Blend selection
# ---------------------------------------------------------------------------

def _best_mcc_over_thresholds(p, y):
    """Oracle MCC over all distinct probability values - selection only."""
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
    iso = IsoCalibrator().fit(p_blend_val, y_val)
    plt = PlattCalibrator().fit(p_blend_val, y_val)
    mcc_iso = _best_mcc_over_thresholds(iso.transform(p_blend_val), y_val)
    mcc_plt = _best_mcc_over_thresholds(plt.transform(p_blend_val), y_val)
    if mcc_iso >= mcc_plt:
        return iso, 'isotonic'
    return plt, 'platt'


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def build_model(X_train, y_train, X_val, y_val):
    """Fit the full Spano 2026 blended architecture; return a picklable bundle.

    On (X_train, y_train): fit XGB and LR-pipe base models.
    On (X_val,   y_val):   fit per-base calibrators, search alpha, select final cal.
    """
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
    y_val_arr   = y_val.values   if hasattr(y_val,   'values') else np.asarray(y_val)

    neg = (y_train_arr == 0).sum()
    pos = (y_train_arr == 1).sum()
    scale_pos_weight = float(neg / pos)
    class_weight = {0: 1.0, 1: scale_pos_weight}

    xgb     = fit_xgb(X_train, y_train, scale_pos_weight)
    lr_pipe = fit_lr_pipeline(X_train, y_train, class_weight)

    p_x_raw = xgb.predict_proba(X_val)[:, 1]
    p_l_raw = lr_pipe.predict_proba(X_val)[:, 1]

    cal_x_iso = IsoCalibrator().fit(p_x_raw, y_val_arr)
    cal_x_pl  = PlattCalibrator().fit(p_x_raw, y_val_arr)
    cal_l_iso = IsoCalibrator().fit(p_l_raw, y_val_arr)
    cal_l_pl  = PlattCalibrator().fit(p_l_raw, y_val_arr)

    combos = {
        'iso+iso': (cal_x_iso.transform(p_x_raw), cal_l_iso.transform(p_l_raw)),
        'sig+sig': (cal_x_pl.transform(p_x_raw),  cal_l_pl.transform(p_l_raw)),
    }

    best_combo, best_alpha, best_mcc = 'sig+sig', 0.5, -1.0
    for name, (p_x_cal, p_l_cal) in combos.items():
        alpha, mcc = search_alpha(p_x_cal, p_l_cal, y_val_arr)
        if mcc > best_mcc:
            best_mcc, best_combo, best_alpha = mcc, name, alpha

    p_x_cal_best, p_l_cal_best = combos[best_combo]
    p_blend = best_alpha * p_x_cal_best + (1 - best_alpha) * p_l_cal_best
    final_cal, final_cal_name = select_final_calibrator(p_blend, y_val_arr)

    return {
        'bundle_type':           'blended_xgb_lr_spano2026',
        'xgb':                   xgb,
        'lr_pipe':               lr_pipe,
        'cal_x_iso':             cal_x_iso,
        'cal_x_pl':              cal_x_pl,
        'cal_l_iso':             cal_l_iso,
        'cal_l_pl':              cal_l_pl,
        'alpha':                 best_alpha,
        'which_base_cal':        best_combo,
        'final_calibrator':      final_cal,
        'final_calibrator_name': final_cal_name,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def calibrated_proba(bundle, X):
    """Run inference: base probs → per-base calibrators → alpha blend → final cal."""
    p_x_raw = bundle['xgb'].predict_proba(X)[:, 1]
    p_l_raw = bundle['lr_pipe'].predict_proba(X)[:, 1]
    if bundle['which_base_cal'] == 'iso+iso':
        p_x = bundle['cal_x_iso'].transform(p_x_raw)
        p_l = bundle['cal_l_iso'].transform(p_l_raw)
    else:
        p_x = bundle['cal_x_pl'].transform(p_x_raw)
        p_l = bundle['cal_l_pl'].transform(p_l_raw)
    p_blend = bundle['alpha'] * p_x + (1 - bundle['alpha']) * p_l
    return bundle['final_calibrator'].transform(p_blend)
