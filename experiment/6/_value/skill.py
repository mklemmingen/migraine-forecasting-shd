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


def brier_skill_ci(y_true, y_prob, reference, n_boot: int = 1000, seed: int = 42):
    """Patient-day bootstrap 95% CI on Brier skill. Matches the Methods section's
    patient-day resampling unit; the under-coverage caveat against a patient-cluster
    bootstrap is disclosed in the Methods section.

    Returns a dict with keys: ``estimate``, ``ci_low``, ``ci_high``, ``n_boot``,
    ``n_valid`` (resamples where bs_ref > 0)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    r = np.asarray(reference, dtype=float)
    estimate = brier_skill_score(y, p, r)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs_ref = float(np.mean((r[idx] - y[idx]) ** 2))
        if bs_ref == 0:
            continue
        bs_model = float(np.mean((p[idx] - y[idx]) ** 2))
        vals.append(1.0 - bs_model / bs_ref)
    if not vals:
        return {"estimate": estimate, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": n_boot, "n_valid": 0}
    vals = np.array(vals)
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_boot": n_boot,
        "n_valid": int(vals.size),
    }


def calibration_in_the_large(y_true, y_prob) -> float:
    """Observed-to-expected ratio. CITL = mean(y) / mean(p); 1.0 means the
    average prediction matches the average observed rate. The third leg of the
    Huang 2020 calibration trio (CITL + slope + reliability diagram)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mean_p = float(np.mean(p))
    if mean_p == 0:
        return float("nan")
    return float(np.mean(y) / mean_p)


def citl_ci(y_true, y_prob, n_boot: int = 1000, seed: int = 42):
    """Patient-day bootstrap CI on calibration-in-the-large."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    estimate = calibration_in_the_large(y, p)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mean_p = float(np.mean(p[idx]))
        if mean_p == 0:
            continue
        vals.append(float(np.mean(y[idx]) / mean_p))
    if not vals:
        return {"estimate": estimate, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_boot": n_boot, "n_valid": 0}
    vals = np.array(vals)
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_boot": n_boot,
        "n_valid": int(vals.size),
    }
