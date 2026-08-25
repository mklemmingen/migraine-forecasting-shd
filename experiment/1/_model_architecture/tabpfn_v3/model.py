from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

import torch as _torch

# Prefer a GPU when one is actually usable; fall back to CPU otherwise. The
# previous hardcoded 'cuda' default made every builder raise
# "Torch not compiled with CUDA enabled" on CPU-only machines.
_DEFAULT_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"


def build_tabpfn_v3(X_train, y_train, *, device=_DEFAULT_DEVICE, random_state=0, output_dir=None):
    """Fit TabPFN-v3 (the v3 default classifier checkpoint) and return
    the bare estimator.

    Pinned to ``ModelVersion.V3`` via
    ``TabPFNClassifier.create_default_for_version``. v3 is the package
    default in ``tabpfn>=8.0.0`` and resolves to
    ``tabpfn-v3-classifier-v3_default.ckpt`` from the
    ``Prior-Labs/tabpfn_3`` HuggingFace repository.

    Architecture reworks vs v2.5 / v2.6 (per the TabPFN-3 model report
    and the v8.0.0 release notes - full discussion in docs/tabPfn.MD §5.4):

      - Return to v1-style ICL on per-row embeddings (v2.x used
        alternating row-wise and feature-wise attention; v3 uses a
        two-stage row-compression pipeline before ICL).
      - Attention-based many-class decoder replaces the fixed-width MLP
        head of v2.x; the active ``MAX_NUMBER_OF_CLASSES`` is 160
        (vs 10 in v2.6).
      - RMSNorm replaces LayerNorm throughout.
      - Native missing-value handling via a binary NaN indicator
        concatenated with the cell value before embedding.
      - GPU preprocessing pipeline: ``ENABLE_GPU_PREPROCESSING=True`` is
        the v3 checkpoint default (verifiable via
        ``clf.get_inference_config()``), so quantile normalisation and
        SVD-based feature augmentation execute on GPU as part of the
        forward pass.
      - Feature-subsampling strategies (``auto``, ``balanced``,
        ``random``, ``gini_feature_importance``); v3 defaults to
        ``auto``.
      - Stratified row subsampling that preserves class proportions
        (relevant for the ~5.1% positive-rate target).
      - Auto-scaling of ``n_estimators`` at fit time so every feature is
        covered by at least one ensemble member; the effective count is
        exposed as ``n_estimators_`` after fitting.
      - Hopper-class GPU performance opt-ins: ``torch.compile`` and
        FlashAttention-3. The ``PerformanceOptions.attention_backend``
        default is ``AUTO``, which auto-routes to FA3 on H100-class
        hardware. On the AMD ROCm setup this project runs on, FA3 does
        not activate; PyTorch SDPA is used instead.
      - Larger training corpus from an updated SCM prior with explicit
        many-class, temporal, spatial, and out-of-distribution
        components; v3 is pre-trained exclusively on synthetic data
        (i.e. no real-data continued pre-training as in v2.5_real).

    Defaults inherited from ``create_default_for_version(V3)``:
      - ``n_estimators=8`` and ``softmax_temperature=0.9`` are the v3
        architecture defaults baked into that classmethod (mirrors the
        v2.6 default to keep the cross-version comparison apples-to-apples
        on the ensemble dimension).
      - ``eval_metric=None``, ``balance_probabilities=False``,
        ``tuning_config=None`` - same rationale as the v2.6 builder
        (see ``tabpfn/model.py`` docstring and docs/tabPfn.MD §1 and §4).

    Calibration framing: the TabPFN-3 model report describes v3 as
    having "calibrated predictive distributions in a single forward
    pass" (page 3) but does not report classification ECE / Brier and
    does not compare v3 calibration against v2.x calibration
    empirically. Whether v3's architectural reworks preserve, improve,
    or degrade calibration relative to v2.6 on this dataset is treated
    as an empirical question by this study's ECE10 / Brier columns
    rather than something we pre-assert. See docs/tabPfn.MD §5.4.

    ``random_state`` is threaded from the caller; same rationale as the
    v2.6 builder.

    ``output_dir`` is accepted for builder-contract uniformity (see
    ``_model_architecture/__init__.py``); bare TabPFN-v3 serialises
    fully via joblib and does not need a side directory.
    """
    del output_dir
    base = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        device=device,
        random_state=random_state,
    )
    base.fit(X_train, y_train)
    return base
