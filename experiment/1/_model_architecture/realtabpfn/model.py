from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import prepend_cache_path

# v2.5_real checkpoint filename in the Prior-Labs/tabpfn_2_5 HuggingFace
# repo. Resolved by ModelSource.get_classifier_v2_5().filenames.
_REAL_CHECKPOINT_FILENAME = "tabpfn-v2.5-classifier-v2.5_real.ckpt"


def build_realtabpfn(X_train, y_train, *, device='cuda', random_state=0, output_dir=None):
    """Real-TabPFN-2.5 - TabPFN-v2.5 architecture with the v2.5_real
    checkpoint (continued pre-training on a curated set of real-world
    tabular datasets, per Garg et al. 2025).

    Uses TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
    so the v2.5 architecture defaults (n_estimators=8,
    softmax_temperature=0.9) are inherited, then overrides model_path
    to select the v2.5_real checkpoint instead of v2.5_default.

    Calibration framing is treated as an empirical question by this
    study's ECE10 / Brier columns rather than a pre-asserted property -
    see docs/tabPfn.MD §5.1.

    output_dir is accepted for builder-contract uniformity; this
    variant serialises fully via joblib.
    """
    del output_dir
    base = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5,
        model_path=prepend_cache_path(_REAL_CHECKPOINT_FILENAME),
        device=device,
        random_state=random_state,
    )
    base.fit(X_train, y_train)
    return base
