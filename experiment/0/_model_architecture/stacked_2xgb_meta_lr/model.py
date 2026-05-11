from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBClassifier


def build_stacker(X_train, y_train):
    """Construct and fit the stacked ensemble on the supplied training data.

    Two XGBoost base learners (shallow depth-3 and deep depth-6) stacked via
    an L1-penalised logistic meta-model.  CV folds are kept in chronological
    order (shuffle=False) to respect the time-series structure.

    Extracted so evaluate_cv.py can re-train per fold without duplicating the
    model configuration.
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
        penalty='l1', solver='saga', C=1.0,
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

    Sigmoid (Platt scaling) is preferred over isotonic regression when the
    number of calibration samples is below the ~1000-sample threshold
    """
    probs = stacker.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    # C=1e10 ≈ no regularisation - standard Platt scaling parameterisation
    calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    calibrator.fit(probs, y_cal)
    return calibrator


def calibrated_proba(bundle, X):
    """Run inference through the stacked ensemble + calibrator bundle.

    Handles both LogisticRegression calibrators (predict_proba) and any future
    isotonic-style calibrators (predict) via duck-typing.
    """
    raw = bundle['stacker'].predict_proba(X)[:, 1]
    cal = bundle['calibrator']
    if hasattr(cal, 'predict_proba'):
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    return cal.predict(raw)


def build_model(X_train, y_train, X_val, y_val):
    """Unified API: fit stacker on (X_train, y_train), Platt-calibrate on (X_val, y_val).

    Returned bundle matches the existing on-disk pickle layout so prior
    model.joblib files keep loading after the API addition.
    """
    stacker = build_stacker(X_train, y_train)
    calibrator = fit_sigmoid_calibrator(stacker, X_val, y_val)
    return {'stacker': stacker, 'calibrator': calibrator}
