import os
import sys
from pathlib import Path

import joblib

# Shared imports. _dataRead/ + _eval/ live at experiment/; _seq/ +
# _model_architecture/ live at experiment/<addition>/. _seq is also added
# directly so the pickled SequenceClassifier (module `sklearn_wrapper`,
# referencing `windowing`) reloads in evaluate.py.
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT), str(_ADDITION_ROOT / "_seq")]
from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
from _seq.dataread import load_seq  # noqa: E402
from _model_architecture.tcn.model import build_tcn  # noqa: E402
from _train._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "headache")
TRAIN_PATH = os.path.join(DATA_DIR, "70_15_15", "stratified", "diary_train.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")


def main():
    with capture_training_output(EXPERIMENT_DIR, label='training'):
        print("Loading data (id-bearing X for windowing)...")
        X_train, y_train = load_seq(TRAIN_PATH, loader=select_non_rolling_features)

        print(f"Train set: X={X_train.shape}, y={y_train.shape}")

        print("Fitting sequence (version_tcn) - windows built inside the estimator...")
        model = build_tcn(X_train, y_train, output_dir=EXPERIMENT_DIR)

        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
