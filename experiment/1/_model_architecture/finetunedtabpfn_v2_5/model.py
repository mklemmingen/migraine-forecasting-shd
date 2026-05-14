import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from tabpfn.finetuning import FinetunedTabPFNClassifier

# Pass 1 epoch budget - large enough that the validation curve clearly
# crosses (or hits a long plateau past) the early-stopping point of the
# library default (epochs=30, patience=8). Visualises convergence and
# is there to stop the concern about whether fine-tuning was budget-starved.
_LONG_CYCLE_EPOCHS = 50

# Ensemble size fixed across the three FinetunedTabPFNClassifier phases
# (finetune / validation / final_inference). The library defaults are
# (2, 2, 8), which violates ``use_fixed_preprocessing_seed=True`` (also
# a library default) and emits a UserWarning at construction. Matching
# all three to the base ``TabPFNClassifier`` inference default (8) keeps
# the preprocessing-seed invariant intact AND keeps the inference
# ensemble width comparable with the non-finetuned variants in this
# benchmark (v2-6 / v3-default / v3-binary / v2-5-real), all of which
# call ``TabPFNClassifier(n_estimators=8)`` implicitly via the library
# default. Without this match, finetuned-vs-non-finetuned comparisons
# would conflate the fine-tuning effect with an ensemble-size effect.
_N_ESTIMATORS = 8


class _ConvergenceCapture:
    """FinetuningLogger implementation that records per-epoch metrics for
    later table emission. Implements the four-method protocol required by
    `tabpfn.finetuning.FinetuningLogger`."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.epochs: list[tuple[int, dict[str, float]]] = []

    def setup(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        # Per-step (within-epoch) metrics - not needed for the convergence table.
        pass

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        self.epochs.append((step, dict(metrics)))

    def finish(self) -> None:
        pass


def _emit_convergence_table(out_dir: Path, capture: _ConvergenceCapture,
                             train_shape: tuple[int, int], n_positives: int) -> None:
    """Write the captured per-epoch metrics to a tab-separated text file
    under <leaf>/diagnostics/. The diagnostics directory is separate from
    results/ so the result aggregator does not try to parse these files
    as evaluation metrics."""
    diag_dir = out_dir / 'diagnostics'
    diag_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    uid = str(uuid.uuid4())
    path = diag_dir / f'finetune_convergence_{ts}_{uid}.txt'

    # Discover the union of per-epoch metric keys for the column header
    metric_keys: list[str] = []
    seen: set[str] = set()
    for _, metrics in capture.epochs:
        for k in metrics:
            if k not in seen:
                seen.add(k)
                metric_keys.append(k)

    with open(path, 'w') as f:
        f.write('# Fine-tuning convergence curve (PASS 1 of 2)\n')
        f.write(f'# Generated: {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'# UUID: {uid}\n')
        f.write(f'# Pass 1 config: epochs={_LONG_CYCLE_EPOCHS}, early_stopping=False\n')
        f.write(f'# Pass 1 purpose: record full per-epoch curve to demonstrate\n')
        f.write(f'#                 that the early-stopped model.joblib (PASS 2)\n')
        f.write(f'#                 is at the validation optimum, not the budget edge.\n')
        f.write(f'# Train shape: {train_shape[0]} rows x {train_shape[1]} features\n')
        f.write(f'# Positives:   {n_positives} ({n_positives / train_shape[0] * 100:.1f}%)\n')
        f.write('#\n')
        f.write('# Logger config keys captured:\n')
        for k, v in sorted(capture.config.items()):
            f.write(f'#   {k} = {v}\n')
        f.write('#\n')
        f.write('step\t' + '\t'.join(metric_keys) + '\n')
        for step, metrics in capture.epochs:
            row = [str(step)]
            for k in metric_keys:
                v = metrics.get(k, '')
                row.append(f'{v:.6f}' if isinstance(v, float) else str(v))
            f.write('\t'.join(row) + '\n')


def build_finetunedtabpfn(X_train, y_train, *, device='cuda', random_state=0, output_dir=None):
    """Fine-tuned TabPFN-v2.5 - two-pass design.

    Base model version: ``FinetunedTabPFNClassifier`` in
    ``tabpfn>=7.1.1`` hardcodes ``version=ModelVersion.V2_5`` in its
    internal ``_create_tabpfn_classifier`` (see
    ``tabpfn/finetuning/finetuned_classifier.py``, lines 210-217). The
    folder label ``version_2-5-finetuned`` reflects the actual base
    model; there is no v2.6 anywhere in the fine-tuning pipeline.

    PASS 1 (diagnostic, only when output_dir is provided): full
    `_LONG_CYCLE_EPOCHS` cycle with `early_stopping=False`, captured by a
    `_ConvergenceCapture` logger and emitted to
    `<output_dir>/diagnostics/finetune_convergence_<ts>_<uuid>.txt`.
    Demonstrates that the early-stopped model from PASS 2 sits at the
    validation optimum rather than at the epoch budget - this is to
    stop the concern about whether fine-tuning was budget-starved on
    the AutoTabPFN-vs-Fine-tuned comparison.

    PASS 2 (production): library defaults (`epochs=30`,
    `early_stopping=True`, `patience=8`). The model returned from this
    pass is what evaluate.py loads via joblib - no over-fitting because
    early stopping selects the best validation epoch.

    Ensemble size: ``n_estimators_finetune = n_estimators_validation =
    n_estimators_final_inference = _N_ESTIMATORS`` (8). The library
    defaults (2 / 2 / 8) violate the ``use_fixed_preprocessing_seed``
    invariant (also a library default); matching all three to 8 keeps
    that invariant true AND keeps the inference ensemble comparable
    with the non-finetuned variants, which all run at the
    ``TabPFNClassifier(n_estimators=8)`` library default. See
    ``_N_ESTIMATORS`` block above for the comparability argument.

    Checkpointing: when ``output_dir`` is provided, the library writes
    per-epoch checkpoints to ``<output_dir>/finetune_checkpoints_pass1/``
    (PASS 1, 50 unstopped epochs) and ``<output_dir>/finetune_checkpoints/``
    (PASS 2, production). The directories are gitignored - the production
    model is the joblib pickle in ``<output_dir>/model.joblib``; the
    library checkpoints are only retained to allow mid-fit resume.

    The implementation uses the sklearn-compatible
    `FinetunedTabPFNClassifier` from `tabpfn.finetuning`. All
    hyperparameters not mentioned above are at library defaults
    (learning_rate=1e-5, validation_split_ratio=0.1, eval_metric=None).
    The internal validation split for early stopping is taken from
    X_train and does not interact with the leaf's held-out cal/test
    data.

    Calibration is not pre-asserted for this variant: Hollmann et al.
    (2025) establish TabPFN's calibration property for the in-context-
    learning regime; whether full fine-tuning preserves, improves, or
    degrades that property is treated as an empirical question by this
    study's ECE10 / Brier columns. See docs/tabPfn.MD §5.3.
    """
    common_kwargs = dict(
        device=device,
        random_state=random_state,
        n_estimators_finetune=_N_ESTIMATORS,
        n_estimators_validation=_N_ESTIMATORS,
        n_estimators_final_inference=_N_ESTIMATORS,
    )

    if output_dir is not None:
        out = Path(output_dir)
        capture = _ConvergenceCapture()
        pass1_ckpt = out / 'finetune_checkpoints_pass1'
        pass1_ckpt.mkdir(parents=True, exist_ok=True)
        long_cycle = FinetunedTabPFNClassifier(
            **common_kwargs,
            epochs=_LONG_CYCLE_EPOCHS,
            early_stopping=False,
            experiment_logger=capture,
        )
        # output_dir is a fit() kwarg in FinetunedTabPFNClassifier (the
        # library reads it inside super().fit() to drive checkpoint writes);
        # it is not a constructor kwarg. The library calls
        # `output_dir.mkdir(parents=True, exist_ok=True)` internally
        # (tabpfn/finetuning/finetuned_base.py:539) and does not coerce
        # str -> Path, so the kwarg must be a pathlib.Path, not a string.
        long_cycle.fit(X_train, y_train, output_dir=pass1_ckpt)
        _emit_convergence_table(out, capture, X_train.shape, int(y_train.sum()))
        # PASS 1 model is discarded; we keep only its convergence diagnostic.
        del long_cycle

    pass2_ckpt = (Path(output_dir) / 'finetune_checkpoints') if output_dir is not None else None
    if pass2_ckpt is not None:
        pass2_ckpt.mkdir(parents=True, exist_ok=True)
    base = FinetunedTabPFNClassifier(**common_kwargs)
    base.fit(X_train, y_train, output_dir=pass2_ckpt)
    return base
