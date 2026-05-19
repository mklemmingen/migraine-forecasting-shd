"""HP variant train (2-way ratio): target=migraine, ratio=80_20,
split_type=stratified, tuning_strategy=pareto_AUROC_slope,
tuning_variant=slope_closest.

2-way ratios have no val parquet; chronologically subsplit train into
train_sub (80%) for stacker fit and cal_sub (20%) for the HP search
calibration role. Cal_sub is passed in as "X_val" to the HP machinery,
which then does its own 5-fold CV inside cal_sub for HP scoring (see
docs in hp_model.objective_fn_factory).
"""
import os
import sys
from pathlib import Path

import joblib

# Shared imports
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import load_raw, prep_split, chronological_subsplit  # noqa: E402
from _model_architecture.stacked_2xgb_meta_lr_hp import model as hp_model  # noqa: E402
from _model_architecture.stacked_2xgb_meta_lr_hp import search_lib  # noqa: E402
from _train._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "migraine")
TRAIN_PATH = os.path.join(DATA_DIR, "80_20", "stratified", "diary_train.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

TRAJECTORY_PATH = Path(EXPERIMENT_DIR).parent / "trajectory.jsonl"

TUNING_STRATEGY = "pareto_AUROC_slope"
TUNING_VARIANT  = "slope_closest"
PARETO_X_KEY    = "auroc"
PARETO_Y_KEY    = "slope_dist_to_1"

CAL_RATIO     = 0.20    # train -> train_sub (80%) + cal_sub (20%)
SEARCH_TRIALS = 500
SEARCH_SEED   = 42


def _run_search(X_train_sub, y_train_sub, X_cal_sub, y_cal_sub):
    objective = hp_model.objective_fn_factory(
        X_train_sub, y_train_sub, X_cal_sub, y_cal_sub)
    if TUNING_STRATEGY == "single_AUROC":
        search_lib.run_random_search(
            objective, n_trials=SEARCH_TRIALS,
            trajectory_path=TRAJECTORY_PATH, seed=SEARCH_SEED,
        )
    elif TUNING_STRATEGY == "pareto_AUROC_slope":
        search_lib.run_nsga2_search(
            objective, n_trials=SEARCH_TRIALS,
            trajectory_path=TRAJECTORY_PATH,
            objectives=[("auroc", "maximize"), ("slope_dist_to_1", "minimize")],
            seed=SEARCH_SEED,
        )
    elif TUNING_STRATEGY == "pareto_AUPRC_slope":
        search_lib.run_nsga2_search(
            objective, n_trials=SEARCH_TRIALS,
            trajectory_path=TRAJECTORY_PATH,
            objectives=[("auprc", "maximize"), ("slope_dist_to_1", "minimize")],
            seed=SEARCH_SEED,
        )
    else:
        raise ValueError(f"unknown TUNING_STRATEGY: {TUNING_STRATEGY!r}")


def main():
    with capture_training_output(EXPERIMENT_DIR, label='training'):
        print("Loading train and chronologically subsplitting...")
        df_train_full = load_raw(TRAIN_PATH, loader=None)
        train_sub, cal_sub = chronological_subsplit(df_train_full, cal_ratio=CAL_RATIO)
        X_train_sub, y_train_sub = prep_split(train_sub)
        X_cal_sub,   y_cal_sub   = prep_split(cal_sub)
        print(f"train_sub: {X_train_sub.shape}, cal_sub: {X_cal_sub.shape}")

        existing_trials = (sum(1 for _ in TRAJECTORY_PATH.open())
                           if TRAJECTORY_PATH.exists() else 0)
        if existing_trials < SEARCH_TRIALS:
            if existing_trials > 0:
                print(f"Trajectory has {existing_trials} trials < SEARCH_TRIALS={SEARCH_TRIALS}, rerunning.")
                TRAJECTORY_PATH.unlink()
            print(f"Running {TUNING_STRATEGY} search ({SEARCH_TRIALS} trials)...")
            _run_search(X_train_sub, y_train_sub, X_cal_sub, y_cal_sub)
        else:
            print(f"Reusing existing trajectory ({existing_trials} trials).")

        print(f"Building variant {TUNING_VARIANT!r}...")
        bundle = hp_model.build_model(
            X_train_sub, y_train_sub, X_cal_sub, y_cal_sub,
            variant_key=TUNING_VARIANT,
            trajectory_path=TRAJECTORY_PATH,
            pareto_x_key=PARETO_X_KEY or None,
            pareto_y_key=PARETO_Y_KEY or None,
        )
        joblib.dump(bundle, MODEL_PATH)
        print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
