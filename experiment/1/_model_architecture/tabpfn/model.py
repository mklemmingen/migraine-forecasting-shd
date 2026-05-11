from tabpfn import TabPFNClassifier


def build_tabpfn(X_train, y_train, *, device='cuda', random_state=0, output_dir=None):
    """Fit TabPFN and return the bare estimator.

    Configured per official PriorLabs guidance - see docs/tabPfn.MD for the
    rationale and the calibration-policy asymmetry vs the XGBoost-family
    architectures.

    Defaults left in place deliberately:
      - eval_metric=None: would trigger internal threshold tuning that
        marginally contaminates predict_proba (uses train context as an
        internal val split). The experiment pipeline already selects the
        MCC-optimal operating threshold externally on a held-out cal set,
        so internal tuning would be redundant double-tuning.
      - balance_probabilities=False: setting True directly rewrites the
        output probability vector via class-prior reweighting, which would
        corrupt the Brier Score and ECE10 columns reported in the paper.
      - softmax_temperature=0.9: the constant temperature baked into the
        v2.6 default (TabPFNClassifier.create_default_for_version source).
        With tuning_config=None, no runtime temperature recalibration runs;
        the model's calibration property rests on its meta-trained prior
        (Hollmann et al. 2025) plus this fixed temperature.

    random_state is threaded from the caller so seed sweeps actually
    sweep - TabPFN's library default of 0 silently reuses the same
    8-model ensemble across runs.

    output_dir is accepted for builder-contract uniformity (see
    `_model_architecture/__init__.py`); bare TabPFN serialises fully via
    joblib and does not need a side directory.
    """
    del output_dir
    base = TabPFNClassifier(
        device=device,
        random_state=random_state,
    )
    base.fit(X_train, y_train)
    return base
