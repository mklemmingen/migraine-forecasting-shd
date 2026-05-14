from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import prepend_cache_path

# Binary-specialised v3 classifier checkpoint shipped at
# https://huggingface.co/Prior-Labs/tabpfn_3 . The v3 model card
# describes it verbatim as: "Specialized for binary classification for
# datasets with <200k rows". The card does not elaborate on the
# training distribution beyond that statement, so we cite the wording
# rather than infer the specifics. This study's binary headache /
# migraine target with n ≈ 3.9k rows is inside the <200k-rows regime
# the card specifies.
_BINARY_CHECKPOINT_FILENAME = "tabpfn-v3-classifier-v3_20260417_binary.ckpt"


def build_tabpfn_v3_binary(X_train, y_train, *, device='cuda', random_state=0, output_dir=None):
    """Fit TabPFN-v3 binary-specialised classifier and return the bare
    estimator.

    Pins ``ModelVersion.V3`` via ``create_default_for_version`` for the
    model-instance defaults (``n_estimators=8``,
    ``softmax_temperature=0.9``, ``eval_metric=None``,
    ``balance_probabilities=False``, ``tuning_config=None`` -
    see docs/tabPfn.MD §5.4) and overrides ``model_path`` to select the
    binary-specialised checkpoint instead of the v3 default. Mirrors the
    pattern used by ``realtabpfn/model.py`` for v2.5_real.

    The binary-specialised checkpoint ships with its own baked-in
    ``InferenceConfig`` that differs from the v3 default checkpoint on
    three fields (verifiable via ``clf.get_inference_config()``):
    ``FEATURE_SUBSAMPLING_METHOD`` (binary uses ``'random'``, default
    uses ``'auto'``), ``FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT``
    (binary ``'auto'``, default ``150``), and ``PREPROCESS_TRANSFORMS``
    (different per-estimator transform mixes). ``ENABLE_GPU_PREPROCESSING``
    is ``True`` on both. FlashAttention-3 does not activate on AMD ROCm
    hardware (see docs/tabPfn.MD §5.4).

    Methodological motivation for including this variant: the v3 model
    card recommends starting with the default and treats the specialised
    checkpoints as "useful in ensembling or HPO setups, or tried
    manually in the regime they were trained for". The SHD task fits the
    binary checkpoint's stated regime (binary classification, <200k
    rows), so reporting v3-default and v3-binary side-by-side is the
    empirical answer to whether the binary specialist actually helps on
    this dataset. See docs/tabPfn.MD §5.5 for the framing.

    Calibration framing is treated as an empirical question (same as the
    v3-default variant) - the v3 model report [10] does not single out
    the binary-specialised checkpoint in its experimental results.
    The §4 no-external-Platt argument applies.

    ``random_state`` is threaded from the caller; same rationale as the
    other TabPFN builders.

    ``output_dir`` is accepted for builder-contract uniformity; this
    variant serialises fully via joblib and does not need a side
    directory.
    """
    del output_dir
    base = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        model_path=prepend_cache_path(_BINARY_CHECKPOINT_FILENAME),
        device=device,
        random_state=random_state,
    )
    base.fit(X_train, y_train)
    return base
