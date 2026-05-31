"""Decision curve analysis - net benefit vs threshold probability.

Net benefit weights true positives against false positives by the odds at the
threshold probability, where the threshold encodes the relative harm of a false
positive versus a false negative for the action under consideration
[vickers2006dca, p. 565; p. 567; vickers2019dca, p. 1]. For next-day migraine
the action is pre-emptive acute medication, whose harm trade-off (an unnecessary
dose vs a missed early-treatment window) the threshold parameterises - so the
clinically plausible band is low thresholds.
"""
import numpy as np

# Clinically plausible threshold band for the pre-emptive-medication action: a
# missed attack is costlier than an unnecessary dose, so low thresholds.
# Decision (docs Section 3.1); reported as a curve over this range, not one cut.
DEFAULT_THRESHOLDS = np.linspace(0.01, 0.50, 50)


def net_benefit(y_true, y_prob, thresholds=DEFAULT_THRESHOLDS):
    """Net benefit of the model across threshold probabilities.

    NB(pt) = TP/n - FP/n * pt/(1-pt)   [vickers2006dca, p. 567]
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    out = np.empty(len(thresholds))
    for i, pt in enumerate(thresholds):
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        out[i] = tp / n - (fp / n) * (pt / (1.0 - pt))
    return out


def treat_all_net_benefit(y_true, thresholds=DEFAULT_THRESHOLDS):
    """Net benefit of the 'treat everyone' default policy."""
    prev = float(np.mean(y_true))
    return prev - (1.0 - prev) * (thresholds / (1.0 - thresholds))


def decision_curve(y_true, y_prob, thresholds=DEFAULT_THRESHOLDS) -> dict:
    """Model, treat-all and treat-none (=0) net-benefit curves over thresholds.

    The model is worth acting on only over the threshold range where its curve
    sits above both references [vickers2019dca, p. 4].
    """
    return {
        "thresholds": np.asarray(thresholds),
        "model": net_benefit(y_true, y_prob, thresholds),
        "treat_all": treat_all_net_benefit(y_true, thresholds),
        "treat_none": np.zeros(len(thresholds)),
    }


def decision_curve_ci(y_true, y_prob, patient_ids=None,
                      thresholds=DEFAULT_THRESHOLDS,
                      n_boot: int = 500, seed: int = 42,
                      bootstrap_unit: str = "patient_day") -> dict:
    """Patient-day or patient-cluster bootstrap 95% CI on the net-benefit curve.

    Returns the standard curves plus per-threshold ``model_ci_low`` and
    ``model_ci_high`` arrays. ``bootstrap_unit="patient_day"`` (default) resamples
    rows uniformly with replacement; ``bootstrap_unit="patient_cluster"`` resamples
    patient_id values with replacement and concatenates every row belonging to each
    sampled patient, matching the body §2.8 resampling discipline for the headline
    cells under the §3.5 within-patient serial-dependence finding. Patient-cluster
    requires ``patient_ids`` to be supplied (a 1-D array aligned with y_true)."""
    base = decision_curve(y_true, y_prob, thresholds)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    curves = np.empty((n_boot, len(thresholds)))
    valid = 0

    if bootstrap_unit == "patient_cluster":
        if patient_ids is None:
            raise ValueError("patient_ids required for bootstrap_unit='patient_cluster'")
        pid = np.asarray(patient_ids)
        unique = np.unique(pid)
        groups = {u: np.where(pid == u)[0] for u in unique}
        for k in range(n_boot):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([groups[u] for u in sampled])
            if y[idx].sum() == 0:
                curves[k] = np.nan
                continue
            curves[valid] = net_benefit(y[idx], p[idx], thresholds)
            valid += 1
    elif bootstrap_unit == "patient_day":
        for k in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if y[idx].sum() == 0:
                curves[k] = np.nan
                continue
            curves[valid] = net_benefit(y[idx], p[idx], thresholds)
            valid += 1
    else:
        raise ValueError(f"unknown bootstrap_unit: {bootstrap_unit!r}")

    if valid == 0:
        base["model_ci_low"] = np.full_like(base["model"], np.nan)
        base["model_ci_high"] = np.full_like(base["model"], np.nan)
        base["n_boot_valid"] = 0
        base["bootstrap_unit"] = bootstrap_unit
        return base
    curves = curves[:valid]
    base["model_ci_low"] = np.percentile(curves, 2.5, axis=0)
    base["model_ci_high"] = np.percentile(curves, 97.5, axis=0)
    base["n_boot_valid"] = valid
    base["bootstrap_unit"] = bootstrap_unit
    return base
