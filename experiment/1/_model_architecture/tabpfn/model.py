from tabpfn import TabPFNClassifier


def build_tabpfn(X_train, y_train, *, device='cuda', random_state=0):
    """Fit TabPFN and return the bare estimator.

    Configured per official PriorLabs guidance — see docs/tabPfn.MD for the
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
      - calibrate_temperature=True (default in inference_config): TabPFN's
        in-house posterior-calibration step. Leave enabled.

    random_state is threaded from the caller so seed sweeps actually
    sweep — TabPFN's library default of 0 silently reuses the same
    8-model ensemble across runs.
    """
    base = TabPFNClassifier(
        device=device,
        random_state=random_state,
    )
    base.fit(X_train, y_train)
    return base
