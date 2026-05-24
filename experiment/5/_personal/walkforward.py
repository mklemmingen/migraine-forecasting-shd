"""Per-patient cold-start curve for Addition 5 (RQ4).

How many of a patient's own diary days does personalisation need before it beats
the population baseline for that patient [houle2021bayesian, p. 1264]? Section 9d
showed that on this cohort the personalisation signal is a base-rate effect (the
per-patient random intercept, not the same-day features), so the cold-start lens
is: when does a patient's OWN running attack rate - estimated causally from their
prior days only - start forecasting their next day better than the cohort rate?

For each patient, walking their days in date order, two forecasters predict day
t (which has n_prior = t own days behind it):

  population   : the fixed cohort attack rate (no personalisation).
  personalised : the patient's running attack rate from days [0, t), shrunk
                 toward the cohort rate by a pseudocount alpha (empirical-Bayes;
                 at n_prior=0 it equals the cohort rate, converging to the
                 patient's own rate). Fully causal - only prior own days.

Scored by Brier (proper scoring rule); the cold-start point is the smallest
own-day count at which the personalised Brier drops and stays below the
population Brier. Reads the unsplit per-patient diary (no train/test split is
needed - the personalised forecaster is causal by construction).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

PATIENT_COL = "patient_id"
DATE_COL = "date"


def cold_start_curve(diary: pd.DataFrame, target_col: str = "migraine_target",
                     alpha: float = 5.0) -> pd.DataFrame:
    """Brier of the population vs personalised forecaster by own-days-seen.

    Returns a DataFrame indexed by ``n_prior`` (own days behind the predicted
    day) with columns: n (rows at that depth), brier_population,
    brier_personalised.
    """
    cohort_rate = float(diary[target_col].mean())
    se_pop, se_pers = defaultdict(list), defaultdict(list)
    for _pid, g in diary.groupby(PATIENT_COL, sort=False):
        y = g.sort_values(DATE_COL)[target_col].to_numpy(dtype=float)
        cum_pos = 0.0
        for t in range(len(y)):
            p_pers = (cum_pos + alpha * cohort_rate) / (t + alpha)
            se_pop[t].append((cohort_rate - y[t]) ** 2)
            se_pers[t].append((p_pers - y[t]) ** 2)
            cum_pos += y[t]
    rows = [{"n_prior": n, "n": len(se_pop[n]),
             "brier_population": float(np.mean(se_pop[n])),
             "brier_personalised": float(np.mean(se_pers[n]))}
            for n in sorted(se_pop)]
    return pd.DataFrame(rows)


def binned_curve(curve: pd.DataFrame, edges=(0, 1, 4, 8, 15, 30, 10_000)) -> pd.DataFrame:
    """Row-weighted Brier per own-day band (stabilises the tail where few
    patients have many days). ``edges`` are right-open bin boundaries."""
    labels = [f"{lo}-{hi-1}" if hi < 10_000 else f"{lo}+"
              for lo, hi in zip(edges[:-1], edges[1:])]
    band = pd.cut(curve["n_prior"], bins=edges, right=False, labels=labels)
    out = []
    for lab, g in curve.groupby(band, observed=True):
        w = g["n"].to_numpy()
        out.append({"own_days": lab, "rows": int(w.sum()),
                    "brier_population": float(np.average(g["brier_population"], weights=w)),
                    "brier_personalised": float(np.average(g["brier_personalised"], weights=w))})
    res = pd.DataFrame(out)
    res["personalised_helps"] = res["brier_personalised"] < res["brier_population"]
    return res


def cold_start_point(curve: pd.DataFrame, min_n: int = 30, window: int = 3) -> int:
    """Smallest n_prior with >= min_n rows where the personalised Brier is below
    the population Brier and stays below over the next ``window`` depths. Returns
    -1 if personalisation never sustainably helps.
    """
    c = curve[curve["n"] >= min_n].reset_index(drop=True)
    better = (c["brier_personalised"] < c["brier_population"]).to_numpy()
    for i in range(len(c) - window + 1):
        if better[i:i + window].all():
            return int(c.loc[i, "n_prior"])
    return -1
