"""Forecast skill - Brier skill score against a per-patient climatology.

A model can have quality (good discrimination/calibration) yet no value beyond
the trivial baseline of each patient's own base rate; skill measures the
increment over that climatological forecast [murphy1993forecast, p. 281]. The
Brier score is the proper scoring rule it is built on [huang2020calibration,
p. 624].
"""
import numpy as np


def brier(y_true, y_prob) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def per_patient_climatology(patient_id, base_rate_by_patient, cohort_rate):
    """Reference forecast = each row's patient base rate.

    Decision (docs Section 3.2): the climatology is each patient's base rate
    estimated from TRAINING data (not the test labels, which would peek);
    patients unseen in train fall back to the cohort training rate. The driver
    passes ``base_rate_by_patient`` (a dict from the train split) and
    ``cohort_rate``.
    """
    return np.array([base_rate_by_patient.get(pid, cohort_rate) for pid in patient_id],
                    dtype=float)


def brier_skill_score(y_true, y_prob, reference) -> float:
    """1 - BS_model / BS_reference. > 0 means the model beats climatology;
    ~0 with a high AUROC exposes discrimination that adds no probabilistic value
    over the base rate [murphy1993forecast, p. 281]."""
    bs_model = brier(y_true, y_prob)
    bs_ref = brier(y_true, reference)
    return float("nan") if bs_ref == 0 else 1.0 - bs_model / bs_ref
