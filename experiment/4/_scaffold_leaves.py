"""
_scaffold_leaves.py - Generate train.py / evaluate.py for addition-4 leaves
(sequence-model family), in the same per-leaf style as Additions 0 and 1.

Run: `.venv/bin/python experiment/4/_scaffold_leaves.py [--force]`

Addition 4 is a model-benchmarking addition: one fitted model per config, each
leaf emitting a results_*.txt in the sharedMetricPrinter contract that
run_aggregate_results.py folds into comparison_*.html alongside the Additions
0/1 tabular cells. That is why it follows the leaf-scaffold pattern rather than
the Addition 3 driver pattern (Addition 3 fits no models). See
docs/addition4_sequence.md.

Layout
------
  experiment/4/<target>/<feature_set>/sequence/version_<arch>/<ratio>/<split>/
                       {train,evaluate}.py

The architecture is the version axis (as in Addition 1): one arch_dir,
"sequence", with one version per nn.Module factory. Each version maps to a
module under _model_architecture/ and a build_<arch> function - see VERSIONS.

Scope (intentionally lean; widen by editing enumerate_leaves)
------------------------------------------------------------
- ratio 70_15_15 only: a real validation split for operating-threshold
  selection, matching the canonical chronological cell the comparison uses.
- splits chrono (the honest forecast) and stratified (the leakage contrast,
  docs/results_findings.md Sec. 1). No patient split here - whole-patient
  generalisation belongs to Addition 5 (personalisation).
- evaluate_cv.py (expanding-window TimeSeriesSplit) is generated for the chrono
  cells: it is Addition 4's PRIMARY internal-validation estimate, because on
  ~201 positive migraine days a single 15% hold-out window is too small and
  time-period-dependent for a stable point estimate. The single hold-out
  (evaluate.py) is the locked secondary check. Stratified cells get the hold-out
  only - TimeSeriesSplit CV is the temporal arm; the stratified split is the
  leakage-contrast hold-out. See docs/addition4_sequence.md Section 3.4.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

ADDITION_ROOT   = Path(__file__).resolve().parent    # experiment/4/
ADDITION        = ADDITION_ROOT.name                 # "4"
EXPERIMENT_ROOT = ADDITION_ROOT.parent               # experiment/
_TEMPLATES_DIR  = ADDITION_ROOT / "_templates"

ARCH_DIR = "sequence"   # architecture-family folder; constant for this addition


class Version(NamedTuple):
    label: str        # folder name -> "version_<label>"
    module: str       # _model_architecture/<module>/model.py
    build_fn: str     # callable imported from that module


VERSIONS: tuple[Version, ...] = (
    Version("gru",        "gru",        "build_gru"),
    Version("tcn",        "tcn",        "build_tcn"),
    Version("window-mlp", "window_mlp", "build_window_mlp"),
)


class Leaf(NamedTuple):
    target: str           # 'headache' | 'migraine'
    feature_set: str      # 'full_features' | 'no_rolling_features' | 'park_features'
    version: Version
    ratio: str            # '70_15_15'
    split_type: str       # 'chrono' | 'stratified'
    with_cv: bool         # generate evaluate_cv.py? (True on chrono cells)

    @property
    def dir(self) -> Path:
        return (ADDITION_ROOT / self.target / self.feature_set / ARCH_DIR
                / f"version_{self.version.label}" / self.ratio / self.split_type)


def _leaf(target: str, fs: str, version: Version, split_type: str) -> Leaf:
    # CV is the temporal internal-validation arm, so only the chrono cells get
    # it; stratified is the leakage-contrast hold-out only.
    return Leaf(target, fs, version, "70_15_15", split_type, with_cv=(split_type == "chrono"))


def enumerate_leaves() -> list[Leaf]:
    leaves: list[Leaf] = []
    for target in ("headache", "migraine"):
        for fs in ("full_features", "no_rolling_features"):
            for version in VERSIONS:
                for split_type in ("chrono", "stratified"):
                    leaves.append(_leaf(target, fs, version, split_type))
    # park_features: migraine target only (Park's stepwise regression
    # discriminates migraine vs non-migraine headache; see docs/park_features.md).
    for version in VERSIONS:
        for split_type in ("chrono", "stratified"):
            leaves.append(_leaf("migraine", "park_features", version, split_type))
    return leaves


# ---------------------------------------------------------------------------
# Feature-set loader configuration (which whitelist load_seq applies)
# ---------------------------------------------------------------------------

class LoaderCfg(NamedTuple):
    extra_import: Optional[str]   # additional import line, or None
    raw_loader_call: str          # passed to load_seq(..., loader=<this>)


LOADERS = {
    "full_features": LoaderCfg(None, "None"),
    "no_rolling_features": LoaderCfg(
        "from _dataRead.filter_to_no_rolling_features import select_non_rolling_features",
        "select_non_rolling_features",
    ),
    "park_features": LoaderCfg(
        "from _dataRead.filter_to_park_features import select_park_features",
        "select_park_features",
    ),
}


def _extra_imports(loader: LoaderCfg) -> str:
    return f"{loader.extra_import}  # noqa: E402\n" if loader.extra_import else ""


TRAIN_TPL = (_TEMPLATES_DIR / "train.py.tpl").read_text()
EVAL_TPL = (_TEMPLATES_DIR / "evaluate.py.tpl").read_text()
EVAL_CV_TPL = (_TEMPLATES_DIR / "evaluate_cv.py.tpl").read_text()


def render_train(leaf: Leaf) -> str:
    loader = LOADERS[leaf.feature_set]
    return TRAIN_TPL.format(
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
    return EVAL_TPL.format(
        addition=ADDITION,
        version_label=leaf.version.label,
        target=leaf.target,
        feature_set=leaf.feature_set,
        ratio=leaf.ratio,
        split_type=leaf.split_type,
        extra_imports=_extra_imports(loader),
        raw_loader=loader.raw_loader_call,
    )


def render_evaluate_cv(leaf: Leaf) -> str:
    loader = LOADERS[leaf.feature_set]
    return EVAL_CV_TPL.format(
        addition=ADDITION,
        module=leaf.version.module,
        build_fn=leaf.version.build_fn,
        version_label=leaf.version.label,
        target=leaf.target,
        feature_set=leaf.feature_set,
        extra_imports=_extra_imports(loader),
        raw_loader=loader.raw_loader_call,
    )


def write_if_absent(path: Path, content: str, force: bool = False) -> str:
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
    counts: dict[str, int] = {"wrote": 0, "skipped": 0, "forced": 0}
    for leaf in leaves:
        renderers = [("train.py", render_train), ("evaluate.py", render_evaluate)]
        if leaf.with_cv:
            renderers.append(("evaluate_cv.py", render_evaluate_cv))
        for kind, render_fn in renderers:
            target_path = leaf.dir / kind
            status = write_if_absent(target_path, render_fn(leaf), force=force)
            counts[status] = counts.get(status, 0) + 1
            print(f"  [{status:7}] {target_path.relative_to(EXPERIMENT_ROOT)}")
    print("=" * 70)
    print(f"Summary: {counts}")


if __name__ == "__main__":
    import sys as _sys
    main(force="--force" in _sys.argv)
