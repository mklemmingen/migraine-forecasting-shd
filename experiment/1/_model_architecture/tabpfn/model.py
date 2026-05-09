from sklearn.calibration import CalibratedClassifierCV
from tabpfn import TabPFNClassifier


def build_tabpfn(X_train, y_train, X_cal, y_cal, device='cuda'):
    """Fit TabPFN on training data and Platt-calibrate on a separate calibration set.

    Calibration uses a held-out cal set (not the evaluation fold) so that
    the evaluation fold is completely unseen at fit time.

    The device kwarg lets each version wrapper override the compute target:
    pass device='cpu' for environments without a GPU.
    """
    base = TabPFNClassifier(device=device)
    base.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(
        estimator=base,
        method='sigmoid',
        cv='prefit',
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated
