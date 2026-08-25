import gc
import shutil
from pathlib import Path

import torch
from tabpfn_extensions.post_hoc_ensembles import AutoTabPFNClassifier

from ._persistence import AutoTabPFNHandle

import torch as _torch

# Prefer a GPU when one is actually usable; fall back to CPU otherwise. The
# previous hardcoded 'cuda' default made every builder raise
# "Torch not compiled with CUDA enabled" on CPU-only machines.
_DEFAULT_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Per-leaf budget for AutoGluon's preset='best' search. The 'best' preset
# triggers full HPO + larger bagging + meta-learner stacking, so a single
# leaf can use the full ceiling on harder cells. AutoGluon respects the
# budget via early termination; under-runs (the easy cells) finish well
# inside it.
_MAX_TIME_SECONDS = 7200

# AutoGluon training preset. 'best' is the standard benchmark citation:
# full HPO, large ensembles, meta-learner stacking. The library default
# ('medium', triggered when presets is unset) does minimal HPO with a
# smaller ensemble - AutoGluon's own documentation describes it as
# "ideal for initial prototyping", which is the wrong default for a
# benchmark paper. 'extreme' is not used because it requires the
# autogluon.tabular[tabarena] extra and injects TabICL / Mitra / TabDPT /
# TabM as competing base models, breaking the explicit TabPFN-2.5
# scoping that the version_2-5-auto label commits us to.
_PRESETS = 'best'

# AutoGluon meta-learner selection metric. AutoGluon's library default
# is 'accuracy', which is degenerate on imbalanced binary - on the
# migraine target (positive rate ~5%) a constant-negative predictor
# gets ~95% accuracy and "wins" the meta-learner's selection. Setting
# eval_metric='roc_auc' lines the meta-learner's selection criterion
# up with this benchmark's headline metric (AUROC) and the natural
# choice for imbalanced binary classification. AUPRC ('average_precision')
# would weight the positive class more heavily but penalises
# calibration-only models; ROC AUC is the AutoGluon benchmark
# convention and is what TabPFN's own published comparisons report.
_EVAL_METRIC = 'roc_auc'


def build_autotabpfn(X_train, y_train, *, device=_DEFAULT_DEVICE, random_state=0, output_dir=None):
    """AutoTabPFN - post-hoc ensemble of TabPFN-v2.5 configurations
    stacked via AutoGluon as the meta-learner.

    Base model version: ``AutoTabPFNClassifier`` in
    ``tabpfn_extensions==0.3.0`` defaults ``model_version=ModelVersion.V2_5``
    and only supports ``ModelVersion.V2`` and ``ModelVersion.V2_5`` (see
    ``tabpfn_extensions/post_hoc_ensembles/sklearn_interface.py::__init__``,
    line ~393 in the pinned commit). The leaf's captured training output
    confirms this empirically: model names appear as ``RealTabPFN-v2.5_*``
    and ``zip_model_path`` references resolve to
    ``tabpfn-v2.5-classifier-v2.5_*.ckpt``. The folder label
    ``version_2-5-auto`` reflects this; there is no v2.6 anywhere in
    the AutoTabPFN pipeline.

    AutoGluon preset and metric. ``presets='best'`` (see ``_PRESETS``
    above) is the standard AutoGluon benchmark configuration. The
    library default ('medium') does minimal HPO and is documented as
    "ideal for initial prototyping" - the wrong default for a paper.
    ``eval_metric='roc_auc'`` (see ``_EVAL_METRIC``) aligns the
    meta-learner's selection criterion with this benchmark's headline
    metric and avoids the degenerate behaviour of the library default
    ('accuracy'), which on the migraine target's ~5% positive rate is
    trivially won by a constant-negative predictor at ~95% accuracy.
    Both choices are documented in docs/tabPfn.MD §5.2.

    The AutoGluon TabularPredictor inside AutoTabPFNClassifier serialises
    to its own on-disk directory; the joblib pickle of the wrapper holds
    only a path reference to that directory. To keep the AutoGluon
    predictor co-located with the leaf's ``model.joblib``, the leaf's
    train.py passes its EXPERIMENT_DIR as ``output_dir``, which this
    builder forwards to AutoGluon via ``phe_init_args={'path': ...}``.
    Any non-AutoTabPFN builder ignores ``output_dir`` - see the contract
    in ``_model_architecture/__init__.py``.

    Per-leaf training-time stdout/stderr (including AutoGluon's
    leaderboard, ensemble weights, time-limit warnings, etc.) is
    captured by the generic ``_eval._training_script_output``
    context manager wrapping each leaf's ``train.py`` ``main()``.

    No dedicated peer-reviewed paper documents AutoTabPFN as a named
    classifier; the methodology section must disclose this rather than
    imply otherwise. See docs/tabPfn.MD §5.2 for citation framing.
    """
    phe_init_args = None
    if output_dir is not None:
        predictor_path = Path(output_dir) / 'autogluon_predictor'
        # AutoGluon refuses to silently reuse an existing predictor dir
        # and emits "path already exists! This predictor may overwrite
        # an existing predictor!" at TabularPredictor construction.
        # Removing the dir before construction makes each fit start from
        # a clean state, which is what the benchmark requires - leftover
        # artifacts from a prior fit would otherwise be mixed into the
        # AutoGluon predictor's working tree on the next sweep.
        if predictor_path.exists():
            shutil.rmtree(predictor_path, ignore_errors=True)
        phe_init_args = {'path': str(predictor_path)}
    base = AutoTabPFNClassifier(
        max_time=_MAX_TIME_SECONDS,
        device=device,
        random_state=random_state,
        presets=_PRESETS,
        eval_metric=_EVAL_METRIC,
        phe_init_args=phe_init_args,
    )
    base.fit(X_train, y_train)

    # AutoGluon's TabularPredictor has already been serialised to disk
    # at ``predictor_path`` by the fit() call above. Pickling the
    # ``AutoTabPFNClassifier`` wrapper directly causes a SIGSEGV under
    # PyTorch 2.11.0+rocm7.2 - joblib walks the wrapper's torch tensor
    # references at getstate, and the AMD HIP runtime crashes inside the
    # destructor cascade. The leaf's ``train.py`` calls ``joblib.dump``
    # on whatever the builder returns, so we return an ``AutoTabPFNHandle``
    # instead. The handle stores only the predictor's on-disk path and
    # lazily reloads it at inference time; see ``_persistence.py``.
    if output_dir is None:
        # Caller (a non-leaf smoke or REPL probe) did not give us a
        # predictor directory, so the predictor was written to AutoGluon's
        # default location. Surface the live wrapper anyway and accept
        # the segfault risk for ad-hoc callers.
        return base

    handle = AutoTabPFNHandle(predictor_path)

    # Release GPU references before returning. Without this the
    # ``AutoTabPFNClassifier`` instance is GC'd when the leaf's ``main()``
    # frame unwinds, which can trigger the same segfault during Python
    # interpreter shutdown.
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return handle
