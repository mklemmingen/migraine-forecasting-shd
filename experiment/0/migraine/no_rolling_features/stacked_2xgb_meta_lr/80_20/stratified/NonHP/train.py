import os
import sys
from pathlib import Path

import joblib
import pandas as pd

# Shared imports - _dataRead/ at experiment/, _model_architecture/ at experiment/<addition>/
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import prep_split, chronological_subsplit  # noqa: E402
from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
from _model_architecture.stacked_2xgb_meta_lr.model import build_model  # noqa: E402
from _train._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "migraine")
TRAIN_PATH = os.path.join(DATA_DIR, "80_20", "stratified", "diary_train.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

# 2-way ratio has no val parquet; subsplit train chronologically.
CAL_RATIO = 0.20


def main():
    with capture_training_output(EXPERIMENT_DIR, label='training'):
        print("Loading data...")
        df_train_full = select_non_rolling_features(TRAIN_PATH)
        train_sub, cal_sub = chronological_subsplit(df_train_full, cal_ratio=CAL_RATIO)
        X_train, y_train = prep_split(train_sub)
        X_cal,   y_cal   = prep_split(cal_sub)

        print(f"Train sub: X={X_train.shape}, y={y_train.shape}  (positive rate: {y_train.mean():.3f})")
        print(f"Cal sub:   X={X_cal.shape}, y={y_cal.shape}  (positive rate: {y_cal.mean():.3f})")

        print("Training stacked_2xgb_meta_lr on train_sub; calibrating on cal_sub...")
        bundle = build_model(X_train, y_train, X_cal, y_cal)
        joblib.dump(bundle, MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
