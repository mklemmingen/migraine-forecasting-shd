"""Personalisation regimes + the comparability bridge for Addition 5.

Regimes (docs Section 3.1):
  - pooled        : the existing Additions 0/1/4 model, re-scored here.
  - per_patient   : one model per patient on that patient's own history.
  - partial_pool  : mixed-effects logistic with a patient random intercept,
                    shrinking short/noisy series toward the cohort mean.
  - tabpfn_ctx    : TabPFN per-patient in-context inference (no parameter fit).

``emit_holdout_results`` writes the standard sharedMetricPrinter results file so
each regime folds into the SAME comparison_*.html as Additions 0/1/4 - this is
the comparability bridge. The regimes consistently leave imbalance to the
external threshold step (no reweighting), matching Addition 4 Decision 3
[vandengoorbergh2022imbalance, p. 1525].
"""
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# _eval on sys.path via the driver; metrics_lib is the shared evaluator contract.
from _eval.metrics_lib import find_operating_thresholds, run_bootstrap_evaluation  # noqa: E402


def emit_holdout_results(out_dir, title, y_val, p_val, y_test, p_test) -> str:
    """Write a standard hold-out results_*.txt so the aggregator picks it up.

    out_dir must be the parse_path-compatible leaf directory, i.e.
    experiment/5/<target>/<feature_set>/<regime>/<ratio>/<split>/ ; the file is
    written under out_dir/results/. Reuses the exact thresholds + bootstrap of
    the tabular/sequence evaluators so the metrics are computed identically.
    """
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


# --- regimes (each returns row-aligned test/val probabilities) ----------------

def predict_pooled(model, X_val, X_test):
    """Reuse an existing leaf's fitted model; return (p_val, p_test).

    TODO: load the headline leaf's model.joblib (driver passes it) and call
    predict_proba on the val/test feature matrices. This is the baseline the
    within-person view re-scores - no new training.
    """
    raise NotImplementedError("pooled regime - see TODO")


def fit_per_patient(train_df, val_df, test_df):
    """One model per patient on that patient's own chronological history.

    TODO (decision: estimability floor, docs Section 9): fit only for patients
    with enough own events; patients below the floor fall back to the pooled
    prediction (so every test row still gets a probability). Return row-aligned
    (p_val, p_test). ~20-30 lines.
    """
    raise NotImplementedError("per-patient regime - see TODO")


def fit_partial_pooling(train_df, val_df, test_df):
    """Mixed-effects logistic with a patient random intercept (statsmodels).

    TODO (decision: model spec, docs Section 9): statsmodels BinomialBayesMixedGLM
    or GEE with a patient random intercept (and a random slope on the recent-rate
    term where per-patient events support it), shrinking short series toward the
    cohort mean (docs/addition3_temporal.md Section 9). Return (p_val, p_test).
    No imbalance reweighting [vandengoorbergh2022imbalance, p. 1525]. ~25-35 lines.
    """
    raise NotImplementedError("partial-pooling regime - see TODO")


def tabpfn_in_context(train_df, val_df, test_df):
    """Per-patient TabPFN in-context inference (the patient's own prior days are
    the in-context training set at prediction time; no parameter fit).

    TODO (optional regime): reuse the Addition 1 TabPFN builder; for each test
    row, condition on that patient's prior days. Cheap once the model is loaded.
    """
    raise NotImplementedError("tabpfn-in-context regime - see TODO")
