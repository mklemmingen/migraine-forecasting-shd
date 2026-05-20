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
- version_2-6: _model_architecture.tabpfn.build_tabpfn - TabPFN-v2.6
  pinned via ``ModelVersion.V2_6``; no external calibrator (TabPFN is
  meta-trained for calibration; see docs/tabPfn.MD §4 for the rationale).
- version_3-default: _model_architecture.tabpfn_v3.build_tabpfn_v3 -
  TabPFN-v3 (the v3 default classifier checkpoint from the
  Prior-Labs/tabpfn_3 HuggingFace repo); see docs/tabPfn.MD §5.4.
- version_3-binary: _model_architecture.tabpfn_v3_binary.build_tabpfn_v3_binary -
  TabPFN-v3 binary-specialised classifier checkpoint (same v3
  architecture, ``tabpfn-v3-classifier-v3_20260417_binary.ckpt``); see
  docs/tabPfn.MD §5.5 for the rationale.
- version_2-5-real: _model_architecture.realtabpfn.build_realtabpfn -
  Real-TabPFN-2.5 (the v2.5_real checkpoint); see docs/tabPfn.MD §5.1.
- version_2-5-finetuned: _model_architecture.finetunedtabpfn_v2_5.build_finetunedtabpfn -
  fine-tuned TabPFN-v2.5 (``FinetunedTabPFNClassifier`` hardcodes
  ``ModelVersion.V2_5`` internally; the label reflects the actual base
  model, not v2.6); see docs/tabPfn.MD §5.3.
- version_2-5-auto: _model_architecture.autotabpfn_v2_5.build_autotabpfn -
  AutoTabPFN on the v2.5 base (``AutoTabPFNClassifier`` defaults
  ``model_version=ModelVersion.V2_5`` and only supports V2/V2_5; the
  label reflects the actual base, not v2.6). Restricted by
  ``_autotabpfn_filter`` to four promising leaves; see docs/tabPfn.MD §5.2.

Builder contract is documented in _model_architecture/__init__.py.

Two ratio templates and two feature_set loader configs follow the same
shape as experiment/0/_scaffold_leaves.py - see that file's docstring for
the 3-way vs 2-way and per-loader rationale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple, Optional

ADDITION_ROOT   = Path(__file__).resolve().parent    # experiment/<addition>/
ADDITION        = ADDITION_ROOT.name                 # e.g. "1" - derived
EXPERIMENT_ROOT = ADDITION_ROOT.parent               # experiment/
_TEMPLATES_DIR  = ADDITION_ROOT / "_templates"       # leaf-script templates

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
    Version("2-6",           "tabpfn",                "build_tabpfn"),
    Version("3-default",     "tabpfn_v3",             "build_tabpfn_v3"),
    Version("3-binary",      "tabpfn_v3_binary",      "build_tabpfn_v3_binary"),
    Version("2-5-real",      "realtabpfn",            "build_realtabpfn"),
    Version("2-5-finetuned", "finetunedtabpfn_v2_5",  "build_finetunedtabpfn"),
    Version("2-5-auto",      "autotabpfn_v2_5",       "build_autotabpfn", _autotabpfn_filter),
)


# ---------------------------------------------------------------------------
# Leaf enumeration
# ---------------------------------------------------------------------------

class Leaf(NamedTuple):
    target: str             # 'headache' | 'migraine'
    feature_set: str        # 'full_features' | 'no_rolling_features'
    version: Version
    ratio: str              # '70_15_15' | '70_30' | '80_20'
    split_type: str         # 'chrono' (forecast) | 'stratified' (leakage contrast) | 'patient' (generalisation)
    with_cv: bool           # generate evaluate_cv.py? (patient leaves: always False)

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
                    for split_type in ("chrono", "stratified", "patient"):
                        with_cv = (ratio == "70_15_15" and split_type == "chrono")
                        leaf = Leaf(target, fs, version, ratio, split_type, with_cv)
                        if version.leaf_filter is not None and not version.leaf_filter(leaf):
                            continue
                        leaves.append(leaf)
    # park_features: migraine target only. Park et al.'s stepwise multiple
    # logistic regression in Table 4 discriminates migraine vs non-migraine
    # headache; the same trigger-selection rationale does NOT apply to the
    # any-headache target. See docs/park_features.md for the framing.
    for version in VERSIONS:
        for ratio in ("70_15_15", "70_30", "80_20"):
            for split_type in ("chrono", "stratified", "patient"):
                with_cv = (ratio == "70_15_15" and split_type == "chrono")
                leaf = Leaf("migraine", "park_features", version, ratio, split_type, with_cv)
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
        extra_import="from _dataRead.filter_to_no_rolling_features import select_non_rolling_features",
        wrapper=(
            "def load_and_prep_data(filepath):\n"
            '    """No-rolling variant: whitelist same-day flags before (X, y) split."""\n'
            "    return _load_and_prep_data(filepath, loader=select_non_rolling_features)"
        ),
        raw_loader_call="select_non_rolling_features",
    ),
    "park_features": LoaderCfg(
        extra_import="from _dataRead.filter_to_park_features import select_park_features",
        wrapper=(
            "def load_and_prep_data(filepath):\n"
            '    """Park-feature variant: whitelist the 6 Park et al. (2016) stepwise-selected'
            ' triggers (Tab. 4, p. 8) with hormonal_changes derived as menstruation OR ovulation."""\n'
            "    return _load_and_prep_data(filepath, loader=select_park_features)"
        ),
        raw_loader_call="select_park_features",
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

TRAIN_3WAY_TPL = (_TEMPLATES_DIR / "train_3way.py.tpl").read_text()

TRAIN_2WAY_TPL = (_TEMPLATES_DIR / "train_2way.py.tpl").read_text()

EVAL_3WAY_TPL = (_TEMPLATES_DIR / "evaluate_3way.py.tpl").read_text()

EVAL_2WAY_TPL = (_TEMPLATES_DIR / "evaluate_2way.py.tpl").read_text()

EVAL_CV_TPL = (_TEMPLATES_DIR / "evaluate_cv.py.tpl").read_text()


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
