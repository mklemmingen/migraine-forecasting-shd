import os
import sys
from pathlib import Path

import joblib

# Shared imports - _dataRead/ at experiment/, _model_architecture/ at experiment/<addition>/
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import load_and_prep_data as _load_and_prep_data, prep_split  # noqa: E402
from _dataRead.filter_to_no_rolling_features import remove_rolling_features  # noqa: E402
from _model_architecture.finetunedtabpfn.model import build_finetunedtabpfn  # noqa: E402
from _eval._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "headache")
TRAIN_PATH = os.path.join(DATA_DIR, "70_15_15", "chrono", "diary_train.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")


def load_and_prep_data(filepath):
    """No-rolling variant: drop temporal aggregation columns before (X, y) split."""
    return _load_and_prep_data(filepath, loader=remove_rolling_features)


def main():
    with capture_training_output(EXPERIMENT_DIR, label='training'):
        print("Loading data...")
        X_train, y_train = load_and_prep_data(TRAIN_PATH)

        print(f"Train set: X={X_train.shape}, y={y_train.shape}")

        print("Fitting tabpfn (version_2-6-finetuned) on train (no external calibrator - see docs/tabPfn.MD)...")
        model = build_finetunedtabpfn(X_train, y_train, output_dir=EXPERIMENT_DIR)

        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
