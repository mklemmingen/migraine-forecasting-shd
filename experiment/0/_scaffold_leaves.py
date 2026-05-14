"""
_scaffold_leaves.py - Generate train.py / evaluate.py / evaluate_cv.py for
addition-0 leaves: stacked_2xgb_meta_lr (full grid) and blended_xgb_lr_spano2026
(canonical 70_15_15/chrono only).

Run: `.venv/bin/python experiment/0/_scaffold_leaves.py [--force]`

Lives inside the addition it scaffolds for, so it can be copied into a new
addition (1, 2, …) and adapted there without path-rewiring. The addition
number is derived from the script's parent directory name.

Architecture coverage
---------------------
- stacked_2xgb_meta_lr: full (target × feature_set × ratio × split_type) grid
  across all three feature sets (full_features, no_rolling_features,
  spano_features).
- blended_xgb_lr_spano2026: canonical 70_15_15/chrono only. The architecture's
  4-way reuse of the calibration set (per-base iso/Platt, alpha search, final
  cal, threshold) makes threshold-dependent metrics unreliable, so fanning
  out across ratios would add cells that need caveating in any comparison.
  The single canonical slot anchors comparison against the prior bachelor-
  thesis replication; AUROC/AUPRC at this slot remain trustworthy
  (rank-based, calibration-invariant).

Two ratio templates
-------------------
- 3-way (70_15_15): val parquet exists; build_model uses (train, val).
- 2-way (70_30, 80_20): no val parquet; chronologically subsplit train into
  train_sub (80%) for fitting and cal_sub (20%) for the val role.

Three feature_set loader configs
--------------------------------
- full_features: no loader (default pd.read_parquet)
- spano_features: loader = remove_non_spano_features
- no_rolling_features: loader = remove_rolling_features

Both architectures expose the same API: build_model(X_train, y_train, X_val,
y_val) → bundle, and calibrated_proba(bundle, X) → ndarray. Templates are
architecture-agnostic; only the {arch} substring varies between leaves.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

ADDITION_ROOT   = Path(__file__).resolve().parent    # experiment/<addition>/
ADDITION        = ADDITION_ROOT.name                 # e.g. "0" - derived
EXPERIMENT_ROOT = ADDITION_ROOT.parent               # experiment/
_TEMPLATES_DIR  = ADDITION_ROOT / "_templates"       # leaf-script templates

STACKED = "stacked_2xgb_meta_lr"
BLENDED = "blended_xgb_lr_spano2026"


# ---------------------------------------------------------------------------
# Leaf enumeration
# ---------------------------------------------------------------------------

class Leaf(NamedTuple):
    target: str             # 'headache' | 'migraine'
    feature_set: str        # 'full_features' | 'spano_features' | 'no_rolling_features'
    arch: str               # 'stacked_2xgb_meta_lr' | 'blended_xgb_lr_spano2026'
    ratio: str              # '70_15_15' | '70_30' | '80_20'
    split_type: str         # 'chrono' | 'stratified'
    with_cv: bool           # generate evaluate_cv.py?

    @property
    def has_val(self) -> bool:
        return self.ratio == "70_15_15"

    @property
    def dir(self) -> Path:
        return (ADDITION_ROOT / self.target / self.feature_set / self.arch /
                self.ratio / self.split_type / "NonHP")


def enumerate_leaves() -> list[Leaf]:
    leaves: list[Leaf] = []
    for target in ("headache", "migraine"):
        # Stacked: full grid across the three target-agnostic feature sets.
        for fs in ("full_features", "no_rolling_features", "spano_features"):
            for ratio in ("70_15_15", "70_30", "80_20"):
                for split_type in ("chrono", "stratified"):
                    with_cv = (ratio == "70_15_15" and split_type == "chrono")
                    leaves.append(Leaf(target, fs, STACKED, ratio, split_type, with_cv))
        # Blended: canonical anchor only - see module docstring.
        leaves.append(Leaf(target, "spano_features", BLENDED, "70_15_15", "chrono", with_cv=True))
    # park_features: migraine target only. Park et al.'s stepwise multiple
    # logistic regression in Table 4 discriminates migraine vs non-migraine
    # headache; the same trigger-selection rationale does NOT apply to the
    # any-headache target (Park did not run a headache-vs-no-headache
    # stepwise regression). See docs/park_features.md for the framing.
    for ratio in ("70_15_15", "70_30", "80_20"):
        for split_type in ("chrono", "stratified"):
            with_cv = (ratio == "70_15_15" and split_type == "chrono")
            leaves.append(Leaf("migraine", "park_features", STACKED, ratio, split_type, with_cv))
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
    "spano_features": LoaderCfg(
        extra_import="from _dataRead.filter_to_spano_features import select_spano_features",
        wrapper=(
            "def load_and_prep_data(filepath):\n"
            '    """Spano-feature variant: whitelist Spano (2026) columns before (X, y) split."""\n'
            "    return _load_and_prep_data(filepath, loader=select_spano_features)"
        ),
        raw_loader_call="select_spano_features",
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
# Templates - architecture-agnostic via {arch} substitution
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
            arch=leaf.arch,
            target=leaf.target,
            ratio=leaf.ratio,
            split_type=leaf.split_type,
            read_imports=_read_imports(loader),
            extra_imports=_extra_imports(loader),
            wrapper_block=_wrapper_block(loader),
        )
    return TRAIN_2WAY_TPL.format(
        arch=leaf.arch,
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
            arch=leaf.arch,
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
        arch=leaf.arch,
        target=leaf.target,
        feature_set=leaf.feature_set,
        ratio=leaf.ratio,
        split_type=leaf.split_type,
        extra_imports=_extra_imports(loader),
        raw_loader=loader.raw_loader_call,
        loader_kwarg=loader_kwarg,
    )


def render_evaluate_cv(leaf: Leaf) -> str:
    """CV evaluator: load the CV parquet directly with the appropriate loader.

    For full_features, this is plain pd.read_parquet. For spano/no_rolling,
    use the filter function - it reads the parquet and drops the same columns
    the variant excludes from train/test, keeping the CV consistent with the
    leaf's feature set.
    """
    loader = LOADERS[leaf.feature_set]
    extra_imports = (
        f"{loader.extra_import}  # noqa: E402\n"
        if loader.extra_import else ""
    )
    cv_load_call = f"{loader.raw_loader_call}(CV_PATH)"
    return EVAL_CV_TPL.format(
        addition=ADDITION,
        arch=leaf.arch,
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
