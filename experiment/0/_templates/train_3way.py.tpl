import os
import sys
from pathlib import Path

import joblib

# Shared imports - _dataRead/ at experiment/, _model_architecture/ at experiment/<addition>/
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
{read_imports}
{extra_imports}from _model_architecture.{arch_module}.model import build_model  # noqa: E402
from _train._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "{target}")
TRAIN_PATH = os.path.join(DATA_DIR, "{ratio}", "{split_type}", "diary_train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "{ratio}", "{split_type}", "diary_val.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")
{wrapper_block}

def main():
    with capture_training_output(EXPERIMENT_DIR, label='training'):
        print("Loading data...")
        X_train, y_train = load_and_prep_data(TRAIN_PATH)
        X_val, y_val = load_and_prep_data(VAL_PATH)

        print(f"Train set: X={{X_train.shape}}, y={{y_train.shape}}")
        print(f"Val set:   X={{X_val.shape}}, y={{y_val.shape}}")

        print("Training {arch} on (train); calibrating on (val)...")
        bundle = build_model(X_train, y_train, X_val, y_val)
        joblib.dump(bundle, MODEL_PATH)
        print(f"Model saved to: {{MODEL_PATH}}")


if __name__ == "__main__":
    main()
