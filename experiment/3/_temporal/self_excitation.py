"""Discrete-time self-excitation of the daily attack series.

A day-resolution diary cannot support a continuous-time Hawkes process (its
likelihood assumes precise continuous inter-event times); the
scientifically correct analogue is a discrete-time autoregressive logistic
model

    P(attack_t = 1 | history) = sigmoid(b0 + sum_k b_k attack_{t-k}
                                          + same-day trigger terms)

fit on the calendar-regular series so the attack lags are true calendar
lags. Self-excitation appears as positive lag coefficients estimated after
controlling for the same-day triggers, isolating history dependence beyond
what today's triggers explain. A likelihood-ratio test compares the full
model against a triggers-only model. Literature anchor: headache days
cluster, with day 1 predicting day 2 [houle2005timeseries].
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.api as sm

import series as S

DEFAULT_LAGS = (1, 2, 3)
DEFAULT_TRIGGERS = (
    "stress_today", "any_sleep_issue_today",
    "weather_change_today", "overeating_today",
)


def build_design(target, lags=DEFAULT_LAGS, triggers=DEFAULT_TRIGGERS):
    """Pooled design frame: attack_t, calendar attack lags, same-day triggers.

    Attack lags come from the gap-aware reindexed series (so lag k is k
    calendar days); a row survives only when attack_t and every lag are
    observed (no missing day in the window) and the same-day triggers are
    present. Returns (frame, lag_cols, trigger_cols).
    """
    df = S.load_diary(target)
    col = S.attack_column(df)
    sbp = S.build_patient_series(df, col)
    avail = [t for t in triggers if t in df.columns]
    parts = []
    for pid, s in sbp.items():
        frame = pd.DataFrame({"attack": s})
        for k in lags:
            frame[f"lag{k}"] = s.shift(k)
        frame = (frame.dropna(subset=["attack"] + [f"lag{k}" for k in lags])
                 .reset_index().rename(columns={"index": "date"}))
        if frame.empty:
            continue
        sub = df[df[S.PATIENT_COL] == pid][["date"] + avail]
        merged = frame.merge(sub, on="date", how="inner").dropna(subset=avail)
        parts.append(merged)
    design = pd.concat(parts, ignore_index=True)
    return design, [f"lag{k}" for k in lags], avail


def fit_self_excitation(target, lags=DEFAULT_LAGS, triggers=DEFAULT_TRIGGERS):
    """Fit the full (lags + triggers) and reduced (triggers-only) logistic
    models and the likelihood-ratio test of the attack lags.
    """
    design, lag_cols, trig_cols = build_design(target, lags, triggers)
    y = design["attack"].astype(float).to_numpy()
    x_full = sm.add_constant(design[lag_cols + trig_cols].astype(float))
    x_red = sm.add_constant(design[trig_cols].astype(float))
    try:
        full = sm.Logit(y, x_full).fit(disp=0, maxiter=200)
        red = sm.Logit(y, x_red).fit(disp=0, maxiter=200)
    except Exception as exc:  # noqa: BLE001 - report non-convergence honestly
        return {"target": target, "error": f"logit did not converge: {exc}"}

    lr_stat = float(2.0 * (full.llf - red.llf))
    lr_p = float(chi2.sf(lr_stat, df=len(lag_cols)))
    ci = full.conf_int()
    lag_coefs = {
        name: {
            "coef": float(full.params[name]),
            "odds_ratio": float(np.exp(full.params[name])),
            "ci": [float(ci.loc[name, 0]), float(ci.loc[name, 1])],
            "p_value": float(full.pvalues[name]),
        }
        for name in lag_cols
    }
    return {
        "target": target,
        "n_rows": int(len(design)),
        "n_attacks": int(y.sum()),
        "lag_coefs": lag_coefs,
        "trigger_coefs": {
            t: {"coef": float(full.params[t]), "p_value": float(full.pvalues[t])}
            for t in trig_cols
        },
        "lr_stat": lr_stat,
        "lr_df": len(lag_cols),
        "lr_p_value": lr_p,
        "pseudo_r2_full": float(full.prsquared),
        "pseudo_r2_triggers_only": float(red.prsquared),
    }
