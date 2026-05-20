import os
import sys
from pathlib import Path

import joblib

# Shared imports - _dataRead/ at experiment/, _model_architecture/ at experiment/<addition>/
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import load_raw, prep_split  # noqa: E402
from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
from _model_architecture.tabpfn_v3_binary.model import build_tabpfn_v3_binary  # noqa: E402
from _train._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "headache")
TRAIN_PATH = os.path.join(DATA_DIR, "70_30", "patient", "diary_train.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")


def main():
    with capture_training_output(EXPERIMENT_DIR, label='training'):
        print("Loading data...")
        df_train_full = load_raw(TRAIN_PATH, loader=select_non_rolling_features)
        X_train, y_train = prep_split(df_train_full)

        print(f"Train: X={X_train.shape}, y={y_train.shape}  (positive rate: {y_train.mean():.3f})")

        print("Fitting tabpfn (version_3-binary) on full train (no external calibrator - see docs/tabPfn.MD)...")
        model = build_tabpfn_v3_binary(X_train, y_train, output_dir=EXPERIMENT_DIR)

        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model successfully saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
