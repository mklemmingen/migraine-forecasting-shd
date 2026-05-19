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

# Hyperparameter-tuning variants under the HyperparameterTuned/ subtree of
# stacked_2xgb_meta_lr cells. The single_AUROC ladder produces 5 budget
# snapshots (HP020/HP050/HP100/HP200/HP500) from one 500-trial
# RandomSampler trajectory: best-AUROC trial in the first N trials gives
# the params for the HPN cell. Each pareto track produces 3 operating
# points (extreme-x, knee, extreme-y) from one 500-trial NSGA-II
# trajectory in the named 2D objective space.
HP_SINGLE_OBJ_TIERS    = ("HP020", "HP050", "HP100", "HP200", "HP500")
HP_PARETO_AUROC_POINTS = ("auroc_max", "knee", "slope_closest")
HP_PARETO_AUPRC_POINTS = ("auprc_max", "knee", "slope_closest")


# ---------------------------------------------------------------------------
# Leaf enumeration
# ---------------------------------------------------------------------------

class Leaf(NamedTuple):
    target: str                       # 'headache' | 'migraine'
    feature_set: str                  # 'full_features' | 'spano_features' | 'no_rolling_features' | 'park_features'
    arch: str                         # 'stacked_2xgb_meta_lr' | 'blended_xgb_lr_spano2026'
    ratio: str                        # '70_15_15' | '70_30' | '80_20'
    split_type: str                   # 'chrono' | 'stratified'
    with_cv: bool                     # generate evaluate_cv.py?
    tuning_strategy: str = ""         # '' for NonHP, else 'single_AUROC' | 'pareto_AUROC_slope' | 'pareto_AUPRC_slope'
    tuning_variant: str  = ""         # '' for NonHP, else 'HP020' ... 'HP500' | 'auroc_max' | 'knee' | ...

    @property
    def has_val(self) -> bool:
        return self.ratio == "70_15_15"

    @property
    def is_hp(self) -> bool:
        return bool(self.tuning_strategy)

    @property
    def module_arch(self) -> str:
        # Python module path for the builder. HP variants live under the
        # NonHP arch's folder (so the aggregator row-key sort groups them
        # together) but import their builder from the _hp module.
        return f"{self.arch}_hp" if self.is_hp else self.arch

    @property
    def dir(self) -> Path:
        base = (ADDITION_ROOT / self.target / self.feature_set / self.arch /
                self.ratio / self.split_type)
        if self.is_hp:
            return base / "HyperparameterTuned" / self.tuning_strategy / self.tuning_variant
        return base / "NonHP"


# Maps a tuning_strategy to the (pareto_x_key, pareto_y_key) used by the
# variant builder. Single-objective strategies don't use Pareto keys.
HP_STRATEGY_TO_PARETO_KEYS = {
    "single_AUROC":       ("", ""),
    "pareto_AUROC_slope": ("auroc", "slope_dist_to_1"),
    "pareto_AUPRC_slope": ("auprc", "slope_dist_to_1"),
}


def _enumerate_hp_variants(target: str) -> list[tuple[str, str]]:
    """Yield (tuning_strategy, tuning_variant) pairs for a target.

    All targets get the single_AUROC ladder and the pareto_AUROC_slope
    frontier. Migraine cells additionally get the pareto_AUPRC_slope
    frontier; the AUROC/AUPRC cross-check is meaningful only on the
    5% positive-rate target where McDermott et al. (arXiv:2401.06091,
    2024) show the two discrimination metrics can disagree under
    imbalance.
    """
    out: list[tuple[str, str]] = []
    for tier in HP_SINGLE_OBJ_TIERS:
        out.append(("single_AUROC", tier))
    for point in HP_PARETO_AUROC_POINTS:
        out.append(("pareto_AUROC_slope", point))
    if target == "migraine":
        for point in HP_PARETO_AUPRC_POINTS:
            out.append(("pareto_AUPRC_slope", point))
    return out


def enumerate_leaves() -> list[Leaf]:
    leaves: list[Leaf] = []
    for target in ("headache", "migraine"):
        # Stacked NonHP: full grid across the three target-agnostic feature sets.
        for fs in ("full_features", "no_rolling_features", "spano_features"):
            for ratio in ("70_15_15", "70_30", "80_20"):
                for split_type in ("chrono", "stratified"):
                    with_cv = (ratio == "70_15_15" and split_type == "chrono")
                    leaves.append(Leaf(target, fs, STACKED, ratio, split_type, with_cv))
        # Blended NonHP: canonical anchor only - see module docstring.
        leaves.append(Leaf(target, "spano_features", BLENDED, "70_15_15", "chrono", with_cv=True))

        # Hyperparameter-tuned variants: stacked_2xgb_meta_lr x full_features
        # only. full_features is the highest-overfitting-risk feature set
        # (EPV 3.9 on migraine) and therefore the most informative cell to
        # instrument; tuning the lower-EPV feature sets would mostly
        # rediscover the NonHP results at higher compute. blended is
        # excluded because its 4-way cal-set reuse already overfits
        # calibration; tuning compounds that. Same (ratio, split, with_cv)
        # grid as the NonHP cells so HP rows sit as siblings of NonHP rows
        # under one architecture group in the comparison table.
        for ratio in ("70_15_15", "70_30", "80_20"):
            for split_type in ("chrono", "stratified"):
                # CV evaluation is intentionally skipped for HP variants;
                # the standard evaluate_cv template's per-fold refit calls
                # build_model with the NonHP signature. HP cells report
                # hold-out test bootstrap only.
                for tuning_strategy, tuning_variant in _enumerate_hp_variants(target):
                    leaves.append(Leaf(
                        target, "full_features", STACKED, ratio, split_type,
                        False, tuning_strategy, tuning_variant,
                    ))

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
        raw_loader_call="None",
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

TRAIN_HP_3WAY_TPL = (_TEMPLATES_DIR / "train_hp_3way.py.tpl").read_text()

TRAIN_HP_2WAY_TPL = (_TEMPLATES_DIR / "train_hp_2way.py.tpl").read_text()


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def render_train(leaf: Leaf) -> str:
    if leaf.is_hp:
        return render_train_hp(leaf)
    loader = LOADERS[leaf.feature_set]
    if leaf.has_val:
        return TRAIN_3WAY_TPL.format(
            arch=leaf.arch,
            arch_module=leaf.module_arch,
            target=leaf.target,
            ratio=leaf.ratio,
            split_type=leaf.split_type,
            read_imports=_read_imports(loader),
            extra_imports=_extra_imports(loader),
            wrapper_block=_wrapper_block(loader),
        )
    return TRAIN_2WAY_TPL.format(
        arch=leaf.arch,
        arch_module=leaf.module_arch,
        target=leaf.target,
        ratio=leaf.ratio,
        split_type=leaf.split_type,
        extra_imports=_extra_imports(loader),
        raw_loader=loader.raw_loader_call,
    )


def render_train_hp(leaf: Leaf) -> str:
    """Render the train.py for an HP variant. Picks 3-way vs 2-way template
    based on ``leaf.has_val``. HP cells are restricted to full_features so
    the loader is always plain ``pd.read_parquet`` with no extra imports.
    """
    loader = LOADERS[leaf.feature_set]
    pareto_x, pareto_y = HP_STRATEGY_TO_PARETO_KEYS[leaf.tuning_strategy]
    common_args = dict(
        target=leaf.target,
        ratio=leaf.ratio,
        split_type=leaf.split_type,
        tuning_strategy=leaf.tuning_strategy,
        tuning_variant=leaf.tuning_variant,
        pareto_x_key=pareto_x,
        pareto_y_key=pareto_y,
        extra_imports=_extra_imports(loader),
    )
    if leaf.has_val:
        return TRAIN_HP_3WAY_TPL.format(
            read_imports=_read_imports(loader),
            **common_args,
        )
    return TRAIN_HP_2WAY_TPL.format(
        raw_loader=loader.raw_loader_call,
        **common_args,
    )


def render_evaluate(leaf: Leaf) -> str:
    loader = LOADERS[leaf.feature_set]
    if leaf.has_val:
        return EVAL_3WAY_TPL.format(
            addition=ADDITION,
            arch=leaf.arch,
            arch_module=leaf.module_arch,
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
        arch_module=leaf.module_arch,
        target=leaf.target,
        feature_set=leaf.feature_set,
        ratio=leaf.ratio,
        split_type=leaf.split_type,
        extra_imports=_extra_imports(loader),
        raw_loader=loader.raw_loader_call,
        loader_kwarg=loader_kwarg,
    )


def render_evaluate_cv(leaf: Leaf) -> str:
    """CV evaluator: load the CV parquet through load_raw with the
    feature-set's loader callable.

    For full_features, raw_loader_call is "None" which makes load_raw
    default to pd.read_parquet. For spano/no_rolling/park, the bare
    callable name plugs in as the loader kwarg so the same column-filter
    that gates train/test also gates the CV parquet.
    """
    loader = LOADERS[leaf.feature_set]
    extra_imports = (
        f"{loader.extra_import}  # noqa: E402\n"
        if loader.extra_import else ""
    )
    return EVAL_CV_TPL.format(
        addition=ADDITION,
        arch=leaf.arch,
        arch_module=leaf.module_arch,
        target=leaf.target,
        feature_set=leaf.feature_set,
        extra_imports=extra_imports,
        raw_loader=loader.raw_loader_call,
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
    nonhp = [l for l in leaves if not l.is_hp]
    hp    = [l for l in leaves if l.is_hp]
    print(f"Enumerating {len(leaves)} leaves: {len(nonhp)} NonHP, {len(hp)} HP")
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
