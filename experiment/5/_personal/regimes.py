"""Personalisation regimes for Addition 5, with the comparability bridge.

The regimes share one base learner (logistic regression, unweighted - imbalance
is left to the external threshold step, Addition 4 Decision 3
[vandengoorbergh2022imbalance, p. 1525]) and vary only the POOLING, so the
comparison isolates the personalisation effect rather than the architecture:

  pooled        : one global LR fitted on all patients (the baseline).
  per_patient   : one LR per patient on that patient's own history, falling back
                  to the global LR for patients with too few own training rows.
  partial_pool  : the global LR plus an empirical-Bayes per-patient random
                  intercept (Gaussian prior, Newton posterior mode), shrinking
                  short/noisy patient series toward the cohort - the
                  partial-pooling discipline of docs/addition3_temporal.md
                  Section 9, applied to forecasting.

``emit_holdout_results`` writes the standard sharedMetricPrinter results file
under experiment/5/<target>/<feature_set>/<regime>/<ratio>/<split>/results/, so
each regime folds into the SAME comparison_*.html as Additions 0/1/4
(architecture = the regime name).
"""
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _dataRead.read import TARGET_COL  # noqa: E402
from _eval.metrics_lib import find_operating_thresholds, run_bootstrap_evaluation  # noqa: E402

PATIENT_COL = "patient_id"
MIN_PATIENT_TRAIN = 30   # min own training rows for a per-patient model
PARTIAL_POOL_TAU2 = 1.0  # Gaussian prior variance on the per-patient intercept


def _make_lr() -> "object":
    """Unweighted logistic regression with feature standardisation."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def pooled(train, val, test, fc):
    """One global LR on all patients. Returns (p_val, p_test)."""
    lr = _make_lr().fit(train[fc], train[TARGET_COL])
    return lr.predict_proba(val[fc])[:, 1], lr.predict_proba(test[fc])[:, 1]


def per_patient(train, val, test, fc, min_train: int = MIN_PATIENT_TRAIN):
    """Per-patient LR with global fallback for sparse patients."""
    global_lr = _make_lr().fit(train[fc], train[TARGET_COL])
    models = {}
    for pid, g in train.groupby(PATIENT_COL):
        if len(g) >= min_train and g[TARGET_COL].nunique() == 2:
            models[pid] = _make_lr().fit(g[fc], g[TARGET_COL])

    def predict(df):
        p = global_lr.predict_proba(df[fc])[:, 1]
        for pid, idx in df.groupby(PATIENT_COL).indices.items():
            if pid in models:
                p[idx] = models[pid].predict_proba(df.iloc[idx][fc])[:, 1]
        return p

    return predict(val), predict(test)


def _eb_intercept(eta, y, tau2, iters: int = 25) -> float:
    """Posterior mode of a per-patient random intercept u under a Gaussian prior
    N(0, tau2), given the global log-odds ``eta`` and labels ``y``. Newton steps
    on the penalised binomial log-likelihood."""
    u = 0.0
    for _ in range(iters):
        s = _sigmoid(eta + u)
        grad = float(np.sum(y - s)) - u / tau2
        hess = -float(np.sum(s * (1.0 - s))) - 1.0 / tau2
        step = grad / hess
        u -= step
        if abs(step) < 1e-6:
            break
    return u


def partial_pool(train, val, test, fc, tau2: float = PARTIAL_POOL_TAU2):
    """Global LR + empirical-Bayes per-patient random intercept."""
    global_lr = _make_lr().fit(train[fc], train[TARGET_COL])
    eta_tr = global_lr.decision_function(train[fc])
    y_tr = train[TARGET_COL].to_numpy()
    u = {}
    for pid, idx in train.groupby(PATIENT_COL).indices.items():
        u[pid] = _eb_intercept(eta_tr[idx], y_tr[idx], tau2)

    def predict(df):
        eta = global_lr.decision_function(df[fc])
        adj = np.array([u.get(pid, 0.0) for pid in df[PATIENT_COL]])
        return _sigmoid(eta + adj)

    return predict(val), predict(test)


REGIMES = {"pooled": pooled, "per_patient": per_patient, "partial_pool": partial_pool}


def emit_holdout_results(out_dir, title, y_val, p_val, y_test, p_test) -> str:
    """Write a standard hold-out results_*.txt so run_aggregate_results.py folds
    the regime into comparison_*.html. out_dir is the parse_path-compatible leaf
    dir experiment/5/<target>/<feature_set>/<regime>/<ratio>/<split>/."""
    opt_mcc, sens_05 = find_operating_thresholds(y_val, p_val)
    results = run_bootstrap_evaluation(y_test, p_test, opt_mcc, sens_05)
    lines = [
        "=" * 60, title, "=" * 60,
        "Validation Set Derived Thresholds:",
        f" -> MCC-Optimal Threshold:          {opt_mcc:.3f}",
        f" -> Threshold for Sens >= 0.50:     {sens_05:.3f}",
        "-" * 60, f"{'Metric':<25} | Mean [95% CI]", "-" * 60,
    ]
    lines += [f"{m:<25} | {s}" for m, s in results.items()]
    lines.append("=" * 60)
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(results_dir, f"results_{ts}_{uuid.uuid4()}.txt")
    Path(path).write_text("\n".join(lines))
    return path
