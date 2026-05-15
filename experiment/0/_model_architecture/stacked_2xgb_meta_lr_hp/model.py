"""
Parametrised builder for the stacked_2xgb_meta_lr_hp variants.

The variants of this architecture share one Optuna search trajectory per
(cell, search_type) tuple. The search-runner scripts (search_random.py for
the single-objective HP020-HP500 ladder; search_nsga2.py for each Pareto
pair) write ``trajectory.jsonl`` once per cell; each variant then refits
its own stacker from the trajectory-selected hyperparameters via
``build_model`` and saves its own ``model.joblib``.

Public surface
--------------
    build_stacker_with_params(X_train, y_train, params) -> StackingClassifier
    fit_sigmoid_calibrator(stacker, X_cal, y_cal)       -> LogisticRegression
    calibrated_proba(bundle, X)                         -> ndarray
    compute_val_metrics(y_val, y_prob_val)              -> dict[str, float]
    objective_fn_factory(X_train, y_train, X_val, y_val) -> callable
    build_model(X_train, y_train, X_val, y_val, *,
                variant_key, trajectory_path,
                pareto_x_key=None, pareto_y_key=None)   -> dict

The signature of ``build_model`` matches the contract documented in
``experiment/1/_model_architecture/__init__.py`` so the per-variant
``train.py`` and ``evaluate.py`` templates can call it identically to
the NonHP builder.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from _eval.metrics_lib import calibration_slope, expected_calibration_error

from . import search_lib


# ---------------------------------------------------------------------------
# Parametrised stacker construction
# ---------------------------------------------------------------------------

def build_stacker_with_params(X_train, y_train, params: dict) -> StackingClassifier:
    """Stacked XGBoost ensemble with the supplied 5 shared hyperparameters.

    The shallow vs deep max_depth split is held fixed at the values in
    ``search_lib.STRUCTURAL_PARAMS`` so the stacking ensemble retains its
    diverse-depth design intent across the HP search. ``scale_pos_weight``
    is set to the inverse class frequency, matching the NonHP baseline.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    shared = dict(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=int(params["min_child_weight"]),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    structural = search_lib.STRUCTURAL_PARAMS

    base_estimators = [
        ("xgb_shallow", XGBClassifier(max_depth=structural["max_depth_shallow"], **shared)),
        ("xgb_deep",    XGBClassifier(max_depth=structural["max_depth_deep"],    **shared)),
    ]
    meta_model = LogisticRegression(
        penalty="l1", solver="saga", C=1.0,
        class_weight="balanced", random_state=42, max_iter=500,
    )
    stacker = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=KFold(n_splits=5, shuffle=False),
        n_jobs=-1,
    )
    stacker.fit(X_train, y_train)
    return stacker


def fit_sigmoid_calibrator(stacker, X_cal, y_cal) -> LogisticRegression:
    """Platt scaling on the stacker's raw probabilities, identical to the
    NonHP baseline. Kept here rather than imported to avoid coupling to
    ``stacked_2xgb_meta_lr`` (which would tie the two architectures'
    calibration choices together at the import level).
    """
    raw = stacker.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    calibrator.fit(raw, y_cal)
    return calibrator


def calibrated_proba(bundle, X) -> np.ndarray:
    """Run inference through stacker + calibrator. Bundle layout matches
    the NonHP baseline so existing evaluate.py templates work unchanged."""
    raw = bundle["stacker"].predict_proba(X)[:, 1]
    cal = bundle["calibrator"]
    if hasattr(cal, "predict_proba"):
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    return cal.predict(raw)


# ---------------------------------------------------------------------------
# Validation metrics (per-trial objective for the search)
# ---------------------------------------------------------------------------

def compute_val_metrics(y_val, y_prob_val) -> dict:
    """Compute the metric dict for one trial.

    AUROC and AUPRC are rank-based and invariant to calibration, so they
    can be computed on either raw or calibrated probabilities. Brier,
    ECE10, and calibration_slope are calibration-sensitive and are
    computed on the Platt-calibrated probabilities to match what the
    test-set bootstrap will report at evaluate time. slope_dist_to_1 is
    the absolute deviation of the slope from 1.0 (Pareto y-axis).
    """
    slope = float(calibration_slope(y_val, y_prob_val))
    return {
        "auroc":             float(roc_auc_score(y_val, y_prob_val)),
        "auprc":             float(average_precision_score(y_val, y_prob_val)),
        "brier":             float(brier_score_loss(y_val, y_prob_val)),
        "ece10":             float(expected_calibration_error(y_val, y_prob_val)),
        "calibration_slope": slope,
        "slope_dist_to_1":   abs(slope - 1.0),
    }


SEARCH_N_FOLDS = 5  # see objective_fn_factory docstring


_NULL_METRICS = {
    "auroc": 0.5, "auprc": 0.0, "brier": 1.0, "ece10": 1.0,
    "calibration_slope": 0.0, "slope_dist_to_1": 1.0,
}


def _slice(arr, idx):
    """Index a pandas-or-numpy array, preserving the row order."""
    return arr.iloc[idx] if hasattr(arr, "iloc") else arr[idx]


def objective_fn_factory(X_train, y_train, X_val, y_val,
                         n_folds: int = SEARCH_N_FOLDS):
    """Return a (params -> metrics dict) closure for the search runners.

    HP-scoring protocol: 5-fold cross-validation on the calibration
    parquet (``X_val``). For each trial, fit the stacker once on
    ``X_train``, then for each of the ``n_folds`` folds:

      1. Fit Platt on ``X_val[fold_train_idx]``     (4/5 of val)
      2. Score post-Platt metrics on ``X_val[fold_score_idx]``  (1/5)

    The trial's reported metrics are the per-fold average. Per-fold
    metrics are preserved in the trajectory record under the
    ``per_fold`` key for auditability.

    Why CV rather than a single split: a single ``X_val`` used for both
    Platt fitting and scoring makes the post-Platt calibration slope
    collapse toward 1 on that set (Platt's whole job is to enforce
    slope=1 on its fitting data). The K-fold protocol averages over
    folds where calibration data and scoring data are disjoint, so
    slope_dist_to_1 carries a real signal about each configuration's
    calibration generalisability. On the migraine target (5% positive
    rate, ~39 positives in cal_sub) the K-fold average uses all ~39
    positives across folds rather than ~19 in a single 50/50 split,
    cutting calibration-slope standard error by ~30% [Huang 2020].

    NaN-safe: a stacker fit failure returns NULL metrics for all folds.
    A per-fold Platt failure returns NULL metrics for that fold only;
    nan-mean over folds skips the failed folds.

    Leaf agnostic: ``X_val`` is whatever calibration parquet the caller
    has selected. For 3-way (70_15_15) leaves this is the val parquet;
    for 2-way leaves it is cal_sub from
    ``chronological_subsplit(train, cal_ratio=0.20)``. The CV operates
    inside that array regardless of source.

    At evaluate time the variant's bundle is refit on the FULL X_val
    for Platt (see ``build_model``), matching the NonHP protocol so
    cross-architecture comparison is fair.
    """
    from sklearn.model_selection import KFold
    splitter = KFold(n_splits=n_folds, shuffle=False)
    fold_idx_pairs = list(splitter.split(X_val))

    def _eval_one(params: dict) -> dict:
        try:
            stacker = build_stacker_with_params(X_train, y_train, params)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            null = dict(_NULL_METRICS)
            null["per_fold"] = [dict(_NULL_METRICS) for _ in range(n_folds)]
            return null

        per_fold = []
        for calib_idx, score_idx in fold_idx_pairs:
            try:
                X_c, y_c = _slice(X_val, calib_idx), _slice(y_val, calib_idx)
                X_s, y_s = _slice(X_val, score_idx), _slice(y_val, score_idx)
                calibrator = fit_sigmoid_calibrator(stacker, X_c, y_c)
                bundle     = {"stacker": stacker, "calibrator": calibrator}
                y_prob     = calibrated_proba(bundle, X_s)
                per_fold.append(compute_val_metrics(y_s, y_prob))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                per_fold.append(dict(_NULL_METRICS))

        keys = per_fold[0].keys()
        avg = {k: float(np.nanmean([f[k] for f in per_fold])) for k in keys}
        # NaN-safe fallback: if all folds failed, nanmean returns nan; clip
        # to NULL value so downstream scalar consumers don't see nan.
        for k, v in avg.items():
            if math.isnan(v):
                avg[k] = _NULL_METRICS[k]
        avg["per_fold"] = per_fold
        return avg

    return _eval_one


# ---------------------------------------------------------------------------
# Variant -> bundle (the public build_model entry point)
# ---------------------------------------------------------------------------

def build_model(
    X_train, y_train, X_val, y_val,
    *,
    variant_key: str,
    trajectory_path: Path,
    pareto_x_key: str | None = None,
    pareto_y_key: str | None = None,
) -> dict:
    """Build the variant-specific bundle from a search trajectory.

    For single-objective tiers (HP020-HP500), the trajectory is the
    RandomSampler output and the variant's chosen params are the
    best-AUROC among the first ``TIER_TO_TRIAL_CAP[variant_key]`` trials.
    For Pareto operating points, the trajectory is an NSGA-II output and
    ``(pareto_x_key, pareto_y_key)`` declares the 2D objective space in
    which the frontier is taken.

    Raises ``RuntimeError`` when the trajectory does not contain a
    valid record for the requested variant; this surfaces "the search
    has not run yet" loudly rather than silently producing a wrong
    bundle.
    """
    trajectory = search_lib.load_trajectory(trajectory_path)
    if not trajectory:
        raise RuntimeError(
            f"trajectory at {trajectory_path} is empty or missing; "
            "run the parent search_*.py before building this variant"
        )

    if variant_key in search_lib.SINGLE_OBJ_TIERS:
        record = search_lib.select_single_obj_tier(
            trajectory, variant_key, score_key="auroc"
        )
    else:
        if pareto_x_key is None or pareto_y_key is None:
            raise ValueError(
                f"variant {variant_key!r} requires pareto_x_key and "
                "pareto_y_key (e.g. 'auroc' + 'slope_dist_to_1')"
            )
        record = search_lib.select_pareto_point(
            trajectory, variant_key,
            x_key=pareto_x_key, y_key=pareto_y_key,
            x_higher=True, y_higher=False,
        )

    if record is None:
        raise RuntimeError(
            f"no record matches variant_key={variant_key!r} in trajectory at "
            f"{trajectory_path}; trajectory has {len(trajectory)} trials"
        )

    stacker    = build_stacker_with_params(X_train, y_train, record["params"])
    calibrator = fit_sigmoid_calibrator(stacker, X_val, y_val)
    return {"stacker": stacker, "calibrator": calibrator}
