"""Recurrent-event regression on inter-attack gap times.

Each inter-attack interval is a recurrent-event observation: its duration
is the calendar gap to the next attack and the event indicator marks that
the gap ended in an attack (the final interval, from the last attack to the
end of follow-up, is right-censored). The covariate is the trailing attack
rate at the start of the interval (observed attacks in the prior 14 calendar
days), so a hazard ratio above 1 means a higher recent rate shortens the
time to the next attack: history dependence beyond the marginal rate.

Andersen-Gill estimates the overall effect on the event intensity (pooled
intervals, patient-clustered robust SE); PWP-gap-time estimates the effect
conditional on the number of prior events (strata on the event order),
which is the forecasting-relevant quantity [amorim2015recurrent]. Migraine
attacks are sparse, so per-interval counts are low; estimates carry CIs and
the caveat that sparsity widens them.
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import series as S

TRAIL_WINDOW = 14
MAX_ORDER = 4


def _trailing_rate(v, pos, trail=TRAIL_WINDOW):
    """Observed attack rate in the ``trail`` calendar days before ``pos``."""
    window = v[max(0, pos - trail):pos]
    obs = window[~np.isnan(window)]
    return float(obs.mean()) if obs.size else 0.0


def build_intervals(series_by_patient) -> pd.DataFrame:
    """Inter-attack gap-time intervals with event order and trailing rate."""
    rows = []
    for pid, s in series_by_patient.items():
        v = s.to_numpy(dtype=float)
        attack_pos = np.flatnonzero(v == 1.0)
        if attack_pos.size < 2:
            continue
        for order, (a, b) in enumerate(zip(attack_pos[:-1], attack_pos[1:]), start=1):
            rows.append({"patient_id": pid, "gap": int(b - a), "event": 1,
                         "order": min(order, MAX_ORDER),
                         "prior_rate": _trailing_rate(v, a)})
        last = attack_pos[-1]
        tail = int((len(v) - 1) - last)
        if tail > 0:
            rows.append({"patient_id": pid, "gap": tail, "event": 0,
                         "order": min(attack_pos.size, MAX_ORDER),
                         "prior_rate": _trailing_rate(v, last)})
    df = pd.DataFrame(rows)
    return df[df["gap"] > 0].reset_index(drop=True)


def fit_recurrent(target) -> dict:
    """Andersen-Gill (pooled, clustered) and PWP-gap-time (order-stratified)
    Cox models for the trailing-rate effect on the time to the next attack.
    """
    df = S.load_diary(target)
    sbp = S.build_patient_series(df, S.attack_column(df))
    iv = build_intervals(sbp)
    out = {"target": target, "n_intervals": int(len(iv)),
           "n_patients": int(iv["patient_id"].nunique()),
           "n_events": int(iv["event"].sum())}
    try:
        ag = CoxPHFitter().fit(iv[["gap", "event", "prior_rate", "patient_id"]],
                               duration_col="gap", event_col="event",
                               cluster_col="patient_id")
        out["ag"] = {"hr_prior_rate": float(np.exp(ag.params_["prior_rate"])),
                     "p_value": float(ag.summary.loc["prior_rate", "p"])}
    except Exception as exc:  # noqa: BLE001 - report fit failure honestly
        out["ag"] = {"error": str(exc)}
    try:
        pwp = CoxPHFitter().fit(iv[["gap", "event", "prior_rate", "order"]],
                                duration_col="gap", event_col="event",
                                strata=["order"])
        out["pwp"] = {"hr_prior_rate": float(np.exp(pwp.params_["prior_rate"])),
                      "p_value": float(pwp.summary.loc["prior_rate", "p"])}
    except Exception as exc:  # noqa: BLE001
        out["pwp"] = {"error": str(exc)}
    return out
