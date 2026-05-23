"""Operating-point analysis for Addition 6.

Connects the benchmark's existing val-derived MCC threshold to the
decision-analytic view: does the threshold the benchmark already uses for its
threshold-derived metrics coincide with the net-benefit-optimal threshold, and
what sensitivity is achievable at a clinically tolerable false-alarm rate?
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def sensitivity_at_fpr(y_true, y_prob, target_fpr: float = 0.10):
    """Highest sensitivity achievable while keeping FPR <= target_fpr.

    Returns (threshold, sensitivity, fpr). Sweeps a fine threshold grid and
    picks the operating point with the largest recall whose false-positive rate
    stays under the tolerated alarm rate.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    best = (1.0, 0.0, 0.0)
    for t in np.linspace(0.0, 1.0, 201):
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        if fpr <= target_fpr and sens > best[1]:
            best = (float(t), float(sens), float(fpr))
    return best


def net_benefit_optimal_threshold(thresholds, model_nb):
    """Threshold maximising model net benefit over the decision-curve grid."""
    i = int(np.argmax(model_nb))
    return float(thresholds[i]), float(model_nb[i])


def map_threshold(val_mcc_threshold, thresholds, model_nb) -> dict:
    """Compare the benchmark's val-derived MCC threshold to the
    net-benefit-optimal threshold (docs Section 3.3)."""
    nb_t, nb_val = net_benefit_optimal_threshold(thresholds, model_nb)
    return {"val_mcc_threshold": float(val_mcc_threshold),
            "net_benefit_optimal_threshold": nb_t,
            "net_benefit_at_optimal": nb_val}
