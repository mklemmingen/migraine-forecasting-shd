def build_realtabpfn(X_train, y_train, X_cal, y_cal, **kwargs):
    """Fit Real-TabPFN on training data and calibrate on a separate calibration set.

    Signature mirrors build_tabpfn so version wrappers are structurally identical.
    Returned object must expose predict_proba(X) -> array of shape (n, 2).

    TODO: implement once the realtabpfn package API is confirmed.
    """
    raise NotImplementedError("build_realtabpfn is not yet implemented — fill in the Real-TabPFN API here.")
