"""
5-fold time-series cross-validation for tabpfn (version_2-5-real).

Reads diary_cv5_timeseries.parquet (target-level, ratio-independent).
Per fold:
  train_sub (first 80% of training-fold dates)  → fit base model
  cal_sub   (last  20% of training-fold dates)  → fit calibrators + select thresholds
  evaluation (cv_fold == k)                     → score only

Result files: results_cv_<timestamp>_<uuid>.txt
"""
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Shared imports
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import prep_split, chronological_subsplit  # noqa: E402
from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
from _model_architecture.realtabpfn.model import build_realtabpfn  # noqa: E402
from _train._training_script_output import capture_training_output  # noqa: E402
from _eval.metrics_lib import find_operating_thresholds, score_fold  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "headache")
RESULTS_DIR    = os.path.join(EXPERIMENT_DIR, "results")
CV_PATH        = os.path.join(DATA_DIR, "diary_cv5_timeseries.parquet")

N_SPLITS  = 5
CAL_RATIO = 0.20

RESULT_PREFIX = "results_cv"
TITLE         = (
    "STAGE 1 / no_rolling_features / tabpfn (version_2-5-real) - "
    f"{N_SPLITS}-Fold Time-Series CV"
)


def main():
    with capture_training_output(EXPERIMENT_DIR, label='training_cv'):
        _main_inner()


def _main_inner():
    print(f"Loading {CV_PATH} ...")
    cv = select_non_rolling_features(CV_PATH)
    print(f"  Total rows: {len(cv):,}  |  cv_fold distribution: "
          f"{ dict(cv['cv_fold'].value_counts().sort_index()) }")

    fold_metrics    = defaultdict(list)
    fold_thresholds = []
    fold_sizes      = []

    for fold in range(1, N_SPLITS + 1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")

        train_fold = cv[cv['cv_fold'] < fold].copy()
        val_fold   = cv[cv['cv_fold'] == fold].copy()

        train_sub, cal_sub = chronological_subsplit(train_fold, cal_ratio=CAL_RATIO)

        X_train_sub, y_train_sub = prep_split(train_sub)
        X_cal_sub,   y_cal_sub   = prep_split(cal_sub)
        X_val,       y_val       = prep_split(val_fold)

        fold_sizes.append((len(train_sub), len(cal_sub), len(val_fold)))
        print(f"  train_sub: {len(train_sub):>4} rows  |  "
              f"cal_sub: {len(cal_sub):>4} rows  |  "
              f"val: {len(val_fold):>4} rows")
        print(f"  Positive rates - train_sub: {y_train_sub.mean():.3f}  "
              f"cal_sub: {y_cal_sub.mean():.3f}  val: {y_val.mean():.3f}")

        print(f"  Fitting tabpfn on train_sub (no external calibrator - see docs/tabPfn.MD)...")
        model = build_realtabpfn(X_train_sub, y_train_sub, output_dir=EXPERIMENT_DIR)

        p_cal = model.predict_proba(X_cal_sub)[:, 1]
        if len(np.unique(y_cal_sub)) < 2:
            print(f"  WARNING: cal_sub has only one class - using default thresholds.")
            opt_thresh, sens_thresh = 0.50, 0.50
        else:
            opt_thresh, sens_thresh = find_operating_thresholds(y_cal_sub.values, p_cal)
        fold_thresholds.append((opt_thresh, sens_thresh))
        print(f"  Thresholds - MCC-optimal: {opt_thresh:.3f}  Sens>=0.5: {sens_thresh:.3f}")

        p_val = model.predict_proba(X_val)[:, 1]
        scores = score_fold(y_val.values, p_val, opt_thresh, sens_thresh)
        for metric, value in scores.items():
            fold_metrics[metric].append(value)
        print(f"  AUROC: {scores['AUROC']:.3f}  AUPRC: {scores['AUPRC']:.3f}  "
              f"MCC: {scores['MCC (Cal-Optimal)']:.3f}")

    metric_names = list(fold_metrics.keys())
    col_w = 7
    header_folds   = "  ".join(f"F{k:<{col_w-2}}" for k in range(1, N_SPLITS + 1))
    header_summary = f"{'Mean':<{col_w}}  {'Std':<{col_w}}"
    separator = "-" * 60

    output_lines = [
        "=" * 60,
        TITLE,
        "=" * 60,
        f"CV scheme    : expanding-window TimeSeriesSplit, n_splits={N_SPLITS}",
        f"Cal sub-split: last {int(CAL_RATIO*100)}% of each training fold's dates",
        "Thresholds   : selected on cal sub-split - NOT on evaluation fold",
        separator,
        f"{'Metric':<25} | {header_folds} | {header_summary}",
        separator,
    ]

    for metric in metric_names:
        vals = fold_metrics[metric]
        per_fold = "  ".join(f"{v:>{col_w}.3f}" for v in vals)
        mean_str = f"{np.mean(vals):<{col_w}.3f}"
        std_str  = f"{np.std(vals):<{col_w}.3f}"
        output_lines.append(f"{metric:<25} | {per_fold} | {mean_str}  {std_str}")

    output_lines.append(separator)
    output_lines.append("Fold sizes (train_sub / cal_sub / val rows):")
    for i, (n_tr, n_cal, n_v) in enumerate(fold_sizes, 1):
        opt, sens = fold_thresholds[i - 1]
        output_lines.append(
            f"  F{i}: {n_tr:>4} / {n_cal:>4} / {n_v:>4}   "
            f"thresh_mcc={opt:.3f}  thresh_sens={sens:.3f}"
        )
    output_lines.append("=" * 60)

    output_text = "\n".join(output_lines)
    print("\n" + output_text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename  = f"{RESULT_PREFIX}_{timestamp}_{uuid.uuid4()}.txt"
    filepath  = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        f.write(output_text)
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()
