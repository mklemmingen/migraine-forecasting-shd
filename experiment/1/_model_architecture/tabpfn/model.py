from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


def build_tabpfn(X_train, y_train, *, device='cuda', random_state=0, output_dir=None):
    """Fit TabPFN-v2.6 and return the bare estimator.

    Pinned to ``ModelVersion.V2_6`` explicitly via
    ``create_default_for_version`` so the ``version_2-6/`` leaves keep
    using v2.6 regardless of which checkpoint the ``tabpfn`` package
    treats as its current default. Without this pin, a bare
    ``TabPFNClassifier(...)`` call resolves to whichever model is the
    package default at runtime (v8.0.x ships TabPFN-3 as the default),
    which would silently shift the architecture under the
    ``version_2-6`` label across re-runs.

    Configured per official PriorLabs guidance - see docs/tabPfn.MD for
    the rationale and the calibration-policy asymmetry vs the
    XGBoost-family architectures.

    Defaults inherited from ``create_default_for_version(V2_6)``:
      - ``n_estimators=8`` and ``softmax_temperature=0.9`` are the
        v2.6 architecture defaults baked into that classmethod.
      - ``eval_metric=None``: not overridden because internal threshold
        tuning would marginally contaminate ``predict_proba`` (uses train
        context as an internal val split). The experiment pipeline
        already selects the MCC-optimal operating threshold externally on
        a held-out cal set, so internal tuning would be redundant
        double-tuning.
      - ``balance_probabilities=False``: not overridden because setting
        True directly rewrites the output probability vector via
        class-prior reweighting, which would corrupt the Brier Score and
        ECE10 columns reported in the paper.
      - ``tuning_config=None``: no runtime temperature recalibration; the
        model's calibration property rests on its meta-trained prior
        (Hollmann et al. 2025) plus the fixed ``softmax_temperature=0.9``.

    ``random_state`` is threaded from the caller so seed sweeps actually
    sweep - TabPFN's library default of 0 silently reuses the same
    8-model ensemble across runs.

    ``output_dir`` is accepted for builder-contract uniformity (see
    ``_model_architecture/__init__.py``); bare TabPFN serialises fully
    via joblib and does not need a side directory.
    """
    del output_dir
    base = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_6,
        device=device,
        random_state=random_state,
    )
    base.fit(X_train, y_train)
    return base
