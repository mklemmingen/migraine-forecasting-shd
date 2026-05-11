"""
_scaffold_leaves.py - Generate train.py / evaluate.py / evaluate_cv.py for
addition-1 leaves (TabPFN family).

Run: `.venv/bin/python experiment/1/_scaffold_leaves.py [--force]`

Lives inside the addition it scaffolds for, so it can be copied into a new
addition (2, 3, …) and adapted there without path-rewiring. The addition
number is derived from the script's parent directory name.

Layout
------
  experiment/<addition>/<target>/<feature_set>/<arch_dir>/<version_dir>/
                       <ratio>/<split_type>/{train,evaluate,evaluate_cv}.py

Two layout deltas vs experiment/0/_scaffold_leaves.py:
  1. An extra `version_<label>` segment between <arch_dir> and <ratio>, so
     two model variants (e.g., TabPFN v2.6 vs Real-TabPFN) can share one
     architecture-family folder. Each version maps to a different module
     under `_model_architecture/` and a different `build_*` function name -
     see VERSIONS below.
  2. The trailing `NonHP/` segment from addition 0 is omitted. Addition 1
     does not branch on hyperparameter-tuning state; that axis lives in a
     separate addition.

Architecture coverage (current scope)
-------------------------------------
- arch_dir = "tabpfn"
- version_2-6: _model_architecture.tabpfn.build_tabpfn - bare
  TabPFNClassifier, no external calibrator (TabPFN is meta-trained for
  calibration; see docs/tabPfn.MD §4 for the rationale).
- version_Real-TabPFN: defined-but-commented out below - build_realtabpfn
  raises NotImplementedError until the Real-TabPFN API is wired up.
  Uncomment one line in VERSIONS to enable scaffolding for it.

Builder contract is documented in _model_architecture/__init__.py.

Two ratio templates and two feature_set loader configs follow the same
shape as experiment/0/_scaffold_leaves.py - see that file's docstring for
the 3-way vs 2-way and per-loader rationale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple, Optional

ADDITION_ROOT = Path(__file__).resolve().parent      # experiment/<addition>/
ADDITION = ADDITION_ROOT.name                        # e.g. "1" - derived
EXPERIMENT_ROOT = ADDITION_ROOT.parent               # experiment/

ARCH_DIR = "tabpfn"   # architecture-family folder; constant for this addition


# ---------------------------------------------------------------------------
# Versions: each maps a folder label to the (module, build-function) pair.
# ---------------------------------------------------------------------------

class Version(NamedTuple):
    label: str        # used in folder name -> "version_<label>"
    module: str       # _model_architecture/<module>/model.py
    build_fn: str     # callable name imported from that module
    leaf_filter: Optional[Callable[["Leaf"], bool]] = None  # restrict scaffolding


# AutoTabPFN is restricted to the four leaves where TabPFN-v2.6 baseline
# showed the strongest combination of high AUROC AND a non-zero MCC at the
# optimal threshold (i.e. the operating-point comparison is meaningful).
# Selected from `comparison_20260510T200006_1e117745.html`:
#   - migraine / full_features / 70_30 / chrono       (AUROC 0.764, MCC 0.239)
#   - migraine / full_features / 70_15_15 / stratified (AUROC 0.733, MCC 0.205)
#   - headache / full_features / 80_20 / stratified   (AUROC 0.702, MCC 0.162)
#   - headache / full_features / 70_30 / stratified   (AUROC 0.688, MCC 0.229)
# Restricting AutoTabPFN to these four cells reduces the sweep cost from
# ~24 fits × ~2 GPU-hours = 48 GPU-hours to ~8 GPU-hours, while preserving
# the comparison on the cells where post-hoc ensembling has the most upside.
_AUTOTABPFN_PROMISING_LEAVES = frozenset({
    ('migraine', 'full_features', '70_30',    'chrono'),
    ('migraine', 'full_features', '70_15_15', 'stratified'),
    ('headache', 'full_features', '80_20',    'stratified'),
    ('headache', 'full_features', '70_30',    'stratified'),
})


def _autotabpfn_filter(leaf: "Leaf") -> bool:
    return (leaf.target, leaf.feature_set, leaf.ratio, leaf.split_type) in _AUTOTABPFN_PROMISING_LEAVES


VERSIONS: tuple[Version, ...] = (
    Version("2-6",           "tabpfn",          "build_tabpfn"),
    Version("2-5-real",      "realtabpfn",      "build_realtabpfn"),
    Version("2-6-finetuned", "finetunedtabpfn", "build_finetunedtabpfn"),
    Version("2-6-auto",      "autotabpfn",      "build_autotabpfn", _autotabpfn_filter),
)


# ---------------------------------------------------------------------------
# Leaf enumeration
# ---------------------------------------------------------------------------

class Leaf(NamedTuple):
    target: str             # 'headache' | 'migraine'
    feature_set: str        # 'full_features' | 'no_rolling_features'
    version: Version
    ratio: str              # '70_15_15' | '70_30' | '80_20'
    split_type: str         # 'chrono' | 'stratified'
    with_cv: bool           # generate evaluate_cv.py?

    @property
    def has_val(self) -> bool:
        return self.ratio == "70_15_15"

    @property
    def dir(self) -> Path:
        return (ADDITION_ROOT / self.target / self.feature_set / ARCH_DIR
                / f"version_{self.version.label}"
                / self.ratio / self.split_type)


def enumerate_leaves() -> list[Leaf]:
    leaves: list[Leaf] = []
    for target in ("headache", "migraine"):
        for fs in ("full_features", "no_rolling_features"):
            for version in VERSIONS:
                for ratio in ("70_15_15", "70_30", "80_20"):
                    for split_type in ("chrono", "stratified"):
                        with_cv = (ratio == "70_15_15" and split_type == "chrono")
                        leaf = Leaf(target, fs, version, ratio, split_type, with_cv)
                        if version.leaf_filter is not None and not version.leaf_filter(leaf):
                            continue
                        leaves.append(leaf)
    return leaves


# ---------------------------------------------------------------------------
# Loader configuration per feature_set
# ---------------------------------------------------------------------------

class LoaderCfg(NamedTuple):
    extra_import: Optional[str]   # additional `from … import …` line, or None
    wrapper: Optional[str]        # local def load_and_prep_data wrapper, or None
    raw_loader_call: str          # how to load+filter the parquet directly


LOADERS = {
    "full_features": LoaderCfg(
        extra_import=None,
        wrapper=None,
        raw_loader_call="pd.read_parquet",
    ),
    "no_rolling_features": LoaderCfg(
        extra_import="from _dataRead.filter_to_no_rolling_features import remove_rolling_features",
        wrapper=(
            "def load_and_prep_data(filepath):\n"
            '    """No-rolling variant: drop temporal aggregation columns before (X, y) split."""\n'
            "    return _load_and_prep_data(filepath, loader=remove_rolling_features)"
        ),
        raw_loader_call="remove_rolling_features",
    ),
}


def _read_imports(loader: LoaderCfg) -> str:
    if loader.wrapper is None:
        return "from _dataRead.read import load_and_prep_data, prep_split  # noqa: E402"
    return "from _dataRead.read import load_and_prep_data as _load_and_prep_data, prep_split  # noqa: E402"


def _extra_imports(loader: LoaderCfg) -> str:
    return f"{loader.extra_import}  # noqa: E402\n" if loader.extra_import else ""


def _wrapper_block(loader: LoaderCfg) -> str:
    return f"\n\n{loader.wrapper}\n" if loader.wrapper else ""


# ---------------------------------------------------------------------------
# Templates - version-agnostic via {module} + {build_fn} substitution.
# train.py imports build_<fn>, fits, and pickles the estimator.
# evaluate.py loads the pickled estimator and calls .predict_proba directly.
# evaluate_cv.py refits per fold, so it imports build_<fn> too.
# ---------------------------------------------------------------------------

TRAIN_3WAY_TPL = '''\
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
{extra_imports}from _model_architecture.{module}.model import {build_fn}  # noqa: E402
from _eval._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "{target}")
TRAIN_PATH = os.path.join(DATA_DIR, "{ratio}", "{split_type}", "diary_train.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")
{wrapper_block}

def main():
    with capture_training_output(EXPERIMENT_DIR, label='training'):
        print("Loading data...")
        X_train, y_train = load_and_prep_data(TRAIN_PATH)

        print(f"Train set: X={{X_train.shape}}, y={{y_train.shape}}")

        print("Fitting {arch} (version_{version_label}) on train (no external calibrator - see docs/tabPfn.MD)...")
        model = {build_fn}(X_train, y_train, output_dir=EXPERIMENT_DIR)

        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model successfully saved to: {{MODEL_PATH}}")


if __name__ == "__main__":
    main()
'''

TRAIN_2WAY_TPL = '''\
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
from _dataRead.read import prep_split  # noqa: E402
{extra_imports}from _model_architecture.{module}.model import {build_fn}  # noqa: E402
from _eval._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "{target}")
TRAIN_PATH = os.path.join(DATA_DIR, "{ratio}", "{split_type}", "diary_train.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")


def main():
    with capture_training_output(EXPERIMENT_DIR, label='training'):
        print("Loading data...")
        df_train_full = {raw_loader}(TRAIN_PATH)
        X_train, y_train = prep_split(df_train_full)

        print(f"Train: X={{X_train.shape}}, y={{y_train.shape}}  (positive rate: {{y_train.mean():.3f}})")

        print("Fitting {arch} (version_{version_label}) on full train (no external calibrator - see docs/tabPfn.MD)...")
        model = {build_fn}(X_train, y_train, output_dir=EXPERIMENT_DIR)

        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model successfully saved to: {{MODEL_PATH}}")


if __name__ == "__main__":
    main()
'''

# Shared metrics block re-used in both evaluate variants.
METRICS_BLOCK = '''\
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0., 1., n_bins + 1)
    binned = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        mask = (binned == i)
        if mask.sum() > 0:
            ece += np.abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.sum()
    return ece / len(y_true)


def find_operating_thresholds(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    mccs    = [matthews_corrcoef(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    recalls = [recall_score(y_true,      (y_prob >= t).astype(int)) for t in thresholds]
    opt_mcc_thresh = thresholds[np.argmax(mccs)]
    valid = [t for t, r in zip(thresholds, recalls) if r >= 0.50]
    sens_05_thresh = max(valid) if valid else 0.50
    return opt_mcc_thresh, sens_05_thresh


def run_bootstrap_evaluation(y_true, y_prob, opt_mcc_thresh, sens_05_thresh, n_iterations=1000, seed=42):
    np.random.seed(seed)
    y_arr = y_true.values
    metrics = defaultdict(list)
    for _ in range(n_iterations):
        idx = np.random.randint(0, len(y_arr), len(y_arr))
        y_t, y_p = y_arr[idx], y_prob[idx]
        if len(np.unique(y_t)) < 2:
            continue
        metrics['AUROC'].append(roc_auc_score(y_t, y_p))
        metrics['AUPRC'].append(average_precision_score(y_t, y_p))
        metrics['Brier Score'].append(brier_score_loss(y_t, y_p))
        metrics['ECE10'].append(expected_calibration_error(y_t, y_p))
        preds_mcc = (y_p >= opt_mcc_thresh).astype(int)
        metrics['MCC (Optimal)'].append(matthews_corrcoef(y_t, preds_mcc))
        metrics['Sensitivity (>=0.5)'].append(
            recall_score(y_t, (y_p >= sens_05_thresh).astype(int)))
        metrics['Accuracy'].append(accuracy_score(y_t, preds_mcc))
        metrics['Precision'].append(precision_score(y_t, preds_mcc, zero_division=0))
        metrics['Recall'].append(recall_score(y_t, preds_mcc, zero_division=0))
        metrics['F1'].append(f1_score(y_t, preds_mcc, zero_division=0))
    return {
        name: f"{np.mean(v):.3f} [{np.percentile(v, 2.5):.3f} - {np.percentile(v, 97.5):.3f}]"
        for name, v in metrics.items()
    }
'''

EVAL_3WAY_TPL = '''\
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Shared imports
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
{read_imports}
{extra_imports}

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "{target}")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
VAL_PATH   = os.path.join(DATA_DIR, "{ratio}", "{split_type}", "diary_val.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "{ratio}", "{split_type}", "diary_test.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

RESULT_PREFIX = "results"
TITLE         = "STAGE {addition} / {feature_set} / {arch} (version_{version_label})"
{wrapper_block}

{metrics_block}

def main():
    print("Loading datasets and model...")
    X_val,  y_val  = load_and_prep_data(VAL_PATH)
    X_test, y_test = load_and_prep_data(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    y_prob_val  = model.predict_proba(X_val)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    print("Calculating optimal thresholds on Validation set...")
    opt_mcc_thresh, sens_05_thresh = find_operating_thresholds(y_val, y_prob_val)

    print("Running bootstrap evaluation on locked Test set (n=1000)...")
    results = run_bootstrap_evaluation(y_test, y_prob_test, opt_mcc_thresh, sens_05_thresh)

    output_lines = [
        "=" * 60,
        TITLE,
        "=" * 60,
        "Validation Set Derived Thresholds:",
        f" -> MCC-Optimal Threshold:          {{opt_mcc_thresh:.3f}}",
        f" -> Threshold for Sens >= 0.50:     {{sens_05_thresh:.3f}}",
        "-" * 60,
        f"{{'Metric':<25}} | Mean [95% CI]",
        "-" * 60,
    ]
    for metric, result_str in results.items():
        output_lines.append(f"{{metric:<25}} | {{result_str}}")
    output_lines.append("=" * 60)
    output_text = "\\n".join(output_lines)

    print("\\n" + output_text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{{RESULT_PREFIX}}_{{timestamp}}_{{str(uuid.uuid4())}}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        f.write(output_text)
    print(f"\\nResults successfully saved to: {{filepath}}")


if __name__ == "__main__":
    main()
'''

EVAL_2WAY_TPL = '''\
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Shared imports
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import load_and_prep_data, prep_split, chronological_subsplit  # noqa: E402
{extra_imports}

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "{target}")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
TRAIN_PATH = os.path.join(DATA_DIR, "{ratio}", "{split_type}", "diary_train.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "{ratio}", "{split_type}", "diary_test.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "model.joblib")

RESULT_PREFIX = "results"
TITLE         = "STAGE {addition} / {feature_set} / {arch} (version_{version_label})"

# 2-way ratio: no val parquet; reproduce the same chronological subsplit
# of train used during fitting to derive operating thresholds.
CAL_RATIO = 0.20


{metrics_block}

def main():
    print("Loading datasets and model...")
    df_train_full = {raw_loader}(TRAIN_PATH)
    _, cal_sub = chronological_subsplit(df_train_full, cal_ratio=CAL_RATIO)
    X_cal, y_cal = prep_split(cal_sub)
    X_test, y_test = load_and_prep_data(TEST_PATH{loader_kwarg})
    model = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    y_prob_cal  = model.predict_proba(X_cal)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    print("Calculating optimal thresholds on cal sub-split...")
    opt_mcc_thresh, sens_05_thresh = find_operating_thresholds(y_cal, y_prob_cal)

    print("Running bootstrap evaluation on locked Test set (n=1000)...")
    results = run_bootstrap_evaluation(y_test, y_prob_test, opt_mcc_thresh, sens_05_thresh)

    output_lines = [
        "=" * 60,
        TITLE,
        "=" * 60,
        "Cal Sub-Split Derived Thresholds:",
        f" -> MCC-Optimal Threshold:          {{opt_mcc_thresh:.3f}}",
        f" -> Threshold for Sens >= 0.50:     {{sens_05_thresh:.3f}}",
        "-" * 60,
        f"{{'Metric':<25}} | Mean [95% CI]",
        "-" * 60,
    ]
    for metric, result_str in results.items():
        output_lines.append(f"{{metric:<25}} | {{result_str}}")
    output_lines.append("=" * 60)
    output_text = "\\n".join(output_lines)

    print("\\n" + output_text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{{RESULT_PREFIX}}_{{timestamp}}_{{str(uuid.uuid4())}}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        f.write(output_text)
    print(f"\\nResults successfully saved to: {{filepath}}")


if __name__ == "__main__":
    main()
'''

EVAL_CV_TPL = '''\
"""
5-fold time-series cross-validation for {arch} (version_{version_label}).

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Shared imports
_LEAF = Path(__file__).resolve()
_EXP_ROOT = next(p for p in _LEAF.parents if p.name == 'experiment')
_ADDITION_ROOT = next(p for p in _LEAF.parents if p.parent == _EXP_ROOT)
sys.path[0:0] = [str(_EXP_ROOT), str(_ADDITION_ROOT)]
from _dataRead.read import prep_split, chronological_subsplit  # noqa: E402
{extra_imports}from _model_architecture.{module}.model import {build_fn}  # noqa: E402
from _eval._training_script_output import capture_training_output  # noqa: E402

# Configuration
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = str(_EXP_ROOT.parent / "data" / "processed" / "{target}")
RESULTS_DIR    = os.path.join(EXPERIMENT_DIR, "results")
CV_PATH        = os.path.join(DATA_DIR, "diary_cv5_timeseries.parquet")

N_SPLITS  = 5
CAL_RATIO = 0.20

RESULT_PREFIX = "results_cv"
TITLE         = (
    "STAGE {addition} / {feature_set} / {arch} (version_{version_label}) - "
    f"{{N_SPLITS}}-Fold Time-Series CV"
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    binned    = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        mask = binned == i
        if mask.sum() > 0:
            ece += np.abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.sum()
    return ece / len(y_true)


def find_operating_thresholds(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    mccs    = [matthews_corrcoef(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    recalls = [recall_score(y_true,      (y_prob >= t).astype(int)) for t in thresholds]
    opt_mcc_thresh = thresholds[np.argmax(mccs)]
    valid = [t for t, r in zip(thresholds, recalls) if r >= 0.50]
    sens_05_thresh = max(valid) if valid else 0.50
    return opt_mcc_thresh, sens_05_thresh


def score_fold(y_val, p_val, opt_thresh, sens_thresh):
    preds_opt = (p_val >= opt_thresh).astype(int)
    return {{
        'AUROC':               roc_auc_score(y_val, p_val),
        'AUPRC':               average_precision_score(y_val, p_val),
        'Brier Score':         brier_score_loss(y_val, p_val),
        'ECE10':               expected_calibration_error(y_val, p_val),
        'MCC (Cal-Optimal)':   matthews_corrcoef(y_val, preds_opt),
        'Sensitivity (>=0.5)': recall_score(y_val, (p_val >= sens_thresh).astype(int)),
        'Accuracy':            accuracy_score(y_val, preds_opt),
        'Precision':           precision_score(y_val, preds_opt, zero_division=0),
        'Recall':              recall_score(y_val, preds_opt, zero_division=0),
        'F1':                  f1_score(y_val, preds_opt, zero_division=0),
    }}


def main():
    with capture_training_output(EXPERIMENT_DIR, label='training_cv'):
        _main_inner()


def _main_inner():
    print(f"Loading {{CV_PATH}} ...")
    cv = {cv_load_call}
    print(f"  Total rows: {{len(cv):,}}  |  cv_fold distribution: "
          f"{{ dict(cv['cv_fold'].value_counts().sort_index()) }}")

    fold_metrics    = defaultdict(list)
    fold_thresholds = []
    fold_sizes      = []

    for fold in range(1, N_SPLITS + 1):
        print(f"\\n--- Fold {{fold}}/{{N_SPLITS}} ---")

        train_fold = cv[cv['cv_fold'] < fold].copy()
        val_fold   = cv[cv['cv_fold'] == fold].copy()

        train_sub, cal_sub = chronological_subsplit(train_fold, cal_ratio=CAL_RATIO)

        X_train_sub, y_train_sub = prep_split(train_sub)
        X_cal_sub,   y_cal_sub   = prep_split(cal_sub)
        X_val,       y_val       = prep_split(val_fold)

        fold_sizes.append((len(train_sub), len(cal_sub), len(val_fold)))
        print(f"  train_sub: {{len(train_sub):>4}} rows  |  "
              f"cal_sub: {{len(cal_sub):>4}} rows  |  "
              f"val: {{len(val_fold):>4}} rows")
        print(f"  Positive rates - train_sub: {{y_train_sub.mean():.3f}}  "
              f"cal_sub: {{y_cal_sub.mean():.3f}}  val: {{y_val.mean():.3f}}")

        print(f"  Fitting {arch} on train_sub (no external calibrator - see docs/tabPfn.MD)...")
        model = {build_fn}(X_train_sub, y_train_sub, output_dir=EXPERIMENT_DIR)

        p_cal = model.predict_proba(X_cal_sub)[:, 1]
        if len(np.unique(y_cal_sub)) < 2:
            print(f"  WARNING: cal_sub has only one class - using default thresholds.")
            opt_thresh, sens_thresh = 0.50, 0.50
        else:
            opt_thresh, sens_thresh = find_operating_thresholds(y_cal_sub.values, p_cal)
        fold_thresholds.append((opt_thresh, sens_thresh))
        print(f"  Thresholds - MCC-optimal: {{opt_thresh:.3f}}  Sens>=0.5: {{sens_thresh:.3f}}")

        p_val = model.predict_proba(X_val)[:, 1]
        scores = score_fold(y_val.values, p_val, opt_thresh, sens_thresh)
        for metric, value in scores.items():
            fold_metrics[metric].append(value)
        print(f"  AUROC: {{scores['AUROC']:.3f}}  AUPRC: {{scores['AUPRC']:.3f}}  "
              f"MCC: {{scores['MCC (Cal-Optimal)']:.3f}}")

    metric_names = list(fold_metrics.keys())
    col_w = 7
    header_folds   = "  ".join(f"F{{k:<{{col_w-2}}}}" for k in range(1, N_SPLITS + 1))
    header_summary = f"{{'Mean':<{{col_w}}}}  {{'Std':<{{col_w}}}}"
    separator = "-" * 60

    output_lines = [
        "=" * 60,
        TITLE,
        "=" * 60,
        f"CV scheme    : expanding-window TimeSeriesSplit, n_splits={{N_SPLITS}}",
        f"Cal sub-split: last {{int(CAL_RATIO*100)}}% of each training fold's dates",
        "Thresholds   : selected on cal sub-split - NOT on evaluation fold",
        separator,
        f"{{'Metric':<25}} | {{header_folds}} | {{header_summary}}",
        separator,
    ]

    for metric in metric_names:
        vals = fold_metrics[metric]
        per_fold = "  ".join(f"{{v:>{{col_w}}.3f}}" for v in vals)
        mean_str = f"{{np.mean(vals):<{{col_w}}.3f}}"
        std_str  = f"{{np.std(vals):<{{col_w}}.3f}}"
        output_lines.append(f"{{metric:<25}} | {{per_fold}} | {{mean_str}}  {{std_str}}")

    output_lines.append(separator)
    output_lines.append("Fold sizes (train_sub / cal_sub / val rows):")
    for i, (n_tr, n_cal, n_v) in enumerate(fold_sizes, 1):
        opt, sens = fold_thresholds[i - 1]
        output_lines.append(
            f"  F{{i}}: {{n_tr:>4}} / {{n_cal:>4}} / {{n_v:>4}}   "
            f"thresh_mcc={{opt:.3f}}  thresh_sens={{sens:.3f}}"
        )
    output_lines.append("=" * 60)

    output_text = "\\n".join(output_lines)
    print("\\n" + output_text)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename  = f"{{RESULT_PREFIX}}_{{timestamp}}_{{uuid.uuid4()}}.txt"
    filepath  = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        f.write(output_text)
    print(f"\\nResults saved to: {{filepath}}")


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def render_train(leaf: Leaf) -> str:
    loader = LOADERS[leaf.feature_set]
    if leaf.has_val:
        return TRAIN_3WAY_TPL.format(
            arch=ARCH_DIR,
            module=leaf.version.module,
            build_fn=leaf.version.build_fn,
            version_label=leaf.version.label,
            target=leaf.target,
            ratio=leaf.ratio,
            split_type=leaf.split_type,
            read_imports=_read_imports(loader),
            extra_imports=_extra_imports(loader),
            wrapper_block=_wrapper_block(loader),
        )
    return TRAIN_2WAY_TPL.format(
        arch=ARCH_DIR,
        module=leaf.version.module,
        build_fn=leaf.version.build_fn,
        version_label=leaf.version.label,
        target=leaf.target,
        ratio=leaf.ratio,
        split_type=leaf.split_type,
        extra_imports=_extra_imports(loader),
        raw_loader=loader.raw_loader_call,
    )


def render_evaluate(leaf: Leaf) -> str:
    loader = LOADERS[leaf.feature_set]
    if leaf.has_val:
        return EVAL_3WAY_TPL.format(
            addition=ADDITION,
            arch=ARCH_DIR,
            version_label=leaf.version.label,
            target=leaf.target,
            feature_set=leaf.feature_set,
            ratio=leaf.ratio,
            split_type=leaf.split_type,
            read_imports=_read_imports(loader),
            extra_imports=_extra_imports(loader),
            wrapper_block=_wrapper_block(loader),
            metrics_block=METRICS_BLOCK,
        )
    loader_kwarg = (
        f", loader={loader.raw_loader_call}"
        if loader.extra_import is not None else ""
    )
    return EVAL_2WAY_TPL.format(
        addition=ADDITION,
        arch=ARCH_DIR,
        version_label=leaf.version.label,
        target=leaf.target,
        feature_set=leaf.feature_set,
        ratio=leaf.ratio,
        split_type=leaf.split_type,
        extra_imports=_extra_imports(loader),
        raw_loader=loader.raw_loader_call,
        loader_kwarg=loader_kwarg,
        metrics_block=METRICS_BLOCK,
    )


def render_evaluate_cv(leaf: Leaf) -> str:
    """CV evaluator: load the CV parquet via the appropriate loader and refit per fold.

    For full_features the call resolves to plain pd.read_parquet; for
    no_rolling_features the call drops the rolling columns so the CV folds
    are consistent with the leaf's feature set.
    """
    loader = LOADERS[leaf.feature_set]
    extra_imports = (
        f"{loader.extra_import}  # noqa: E402\n"
        if loader.extra_import else ""
    )
    cv_load_call = f"{loader.raw_loader_call}(CV_PATH)"
    return EVAL_CV_TPL.format(
        addition=ADDITION,
        arch=ARCH_DIR,
        module=leaf.version.module,
        build_fn=leaf.version.build_fn,
        version_label=leaf.version.label,
        target=leaf.target,
        feature_set=leaf.feature_set,
        extra_imports=extra_imports,
        cv_load_call=cv_load_call,
    )


def write_if_absent(path: Path, content: str, force: bool = False) -> str:
    """Write content to path; returns 'wrote', 'skipped', or 'forced'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if existed and not force:
        return "skipped"
    path.write_text(content)
    return "forced" if existed else "wrote"


def main(force: bool = False) -> None:
    leaves = enumerate_leaves()
    print(f"Generating {len(leaves)} leaves (force={force})")
    print("=" * 70)

    counts = {"wrote": 0, "skipped": 0, "forced": 0}
    for leaf in leaves:
        for kind, render_fn in (
            ("train.py", render_train),
            ("evaluate.py", render_evaluate),
        ):
            target_path = leaf.dir / kind
            content = render_fn(leaf)
            status = write_if_absent(target_path, content, force=force)
            counts[status] = counts.get(status, 0) + 1
            print(f"  [{status:7}] {target_path.relative_to(EXPERIMENT_ROOT)}")

        if leaf.with_cv:
            cv_path = leaf.dir / "evaluate_cv.py"
            content = render_evaluate_cv(leaf)
            status = write_if_absent(cv_path, content, force=force)
            counts[status] = counts.get(status, 0) + 1
            print(f"  [{status:7}] {cv_path.relative_to(EXPERIMENT_ROOT)}")

    print("=" * 70)
    print(f"Summary: {counts}")


if __name__ == "__main__":
    import sys as _sys
    main(force="--force" in _sys.argv)
