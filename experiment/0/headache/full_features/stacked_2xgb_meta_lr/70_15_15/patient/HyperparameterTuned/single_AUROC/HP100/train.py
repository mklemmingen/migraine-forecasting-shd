"""HP variant train: target=headache, ratio=70_15_15, split_type=patient,
tuning_strategy=single_AUROC, tuning_variant=HP100.

If the parent <strategy>/trajectory.jsonl does not exist, runs the search
(RandomSampler for single_AUROC; NSGA-II for pareto_*) and writes the
trajectory. Then refits this variant's model.joblib from the trajectory-
selected hyperparameters via stacked_2xgb_meta_lr_hp.build_model.
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
from _dataRead.read import load_and_prep_data, prep_split  # noqa: E402
from _model_architecture.stacked_2xgb_meta_lr_hp import model as hp_model  # noqa: E402
from _model_architecture.stacked_2xgb_meta_lr_hp import search_lib  # noqa: E402
from _train._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "headache")
TRAIN_PATH = os.path.join(DATA_DIR, "70_15_15", "patient", "diary_train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "70_15_15", "patient", "diary_val.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

# Trajectory is shared by all variants of this (cell, strategy); the parent
# of this leaf folder is <strategy>/, two levels above the variant folder.
TRAJECTORY_PATH = Path(EXPERIMENT_DIR).parent / "trajectory.jsonl"

TUNING_STRATEGY = "single_AUROC"
TUNING_VARIANT  = "HP100"
PARETO_X_KEY    = ""   # empty string for single-objective
PARETO_Y_KEY    = ""

SEARCH_TRIALS = 500
SEARCH_SEED   = 42


def _run_search(X_train, y_train, X_val, y_val):
    """Populate trajectory.jsonl by running the strategy's search once."""
    objective = hp_model.objective_fn_factory(X_train, y_train, X_val, y_val)
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
        print("Loading datasets...")
        X_train, y_train = load_and_prep_data(TRAIN_PATH)
        X_val,   y_val   = load_and_prep_data(VAL_PATH)

        existing_trials = (sum(1 for _ in TRAJECTORY_PATH.open())
                           if TRAJECTORY_PATH.exists() else 0)
        if existing_trials < SEARCH_TRIALS:
            if existing_trials > 0:
                print(f"Trajectory has {existing_trials} trials < SEARCH_TRIALS={SEARCH_TRIALS}, rerunning.")
                TRAJECTORY_PATH.unlink()
            print(f"Running {TUNING_STRATEGY} search ({SEARCH_TRIALS} trials, seed={SEARCH_SEED})...")
            _run_search(X_train, y_train, X_val, y_val)
        else:
            print(f"Reusing existing trajectory ({existing_trials} trials) at {TRAJECTORY_PATH}")

        print(f"Building variant {TUNING_VARIANT!r}...")
        bundle = hp_model.build_model(
            X_train, y_train, X_val, y_val,
            variant_key=TUNING_VARIANT,
            trajectory_path=TRAJECTORY_PATH,
            pareto_x_key=PARETO_X_KEY or None,
            pareto_y_key=PARETO_Y_KEY or None,
        )
        joblib.dump(bundle, MODEL_PATH)
        print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
