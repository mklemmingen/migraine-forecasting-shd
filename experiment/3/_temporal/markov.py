"""First-order Markov transition analysis of the daily attack series.

P(attack tomorrow | attack today) versus P(attack tomorrow | no attack
today), pooled and per-patient, with a chi-square test of independence and
a comparison to the marginal attack rate. Only consecutive calendar-day
pairs are used (the series module drops gap-straddling pairs), so
"tomorrow" is always the next calendar day. Literature anchor: Markov-chain
attack modelling [barra2020markov]; their migraine-locked day definition
differs from this day-level binary event, which the write-up notes.
"""
import numpy as np
from scipy.stats import chi2_contingency

import series as S


def pooled_transition(series_by_patient) -> dict:
    """Pooled first-order transition contingency and conditional risks."""
    a, b = S.pooled_indicator_pairs(series_by_patient, lag=1)
    if len(a) == 0:
        return {"n_pairs": 0}
    table = np.array([
        [int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())],
        [int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())],
    ])
    row1, row0 = table[1].sum(), table[0].sum()
    p_given_attack = table[1, 1] / row1 if row1 else float("nan")
    p_given_none = table[0, 1] / row0 if row0 else float("nan")
    # chi-square needs all marginals non-zero to be defined.
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        chi2 = p_value = float("nan")
    else:
        chi2, p_value, _dof, _exp = chi2_contingency(table)
    return {
        "table_today_by_tomorrow": table.tolist(),
        "p_attack_tomorrow_given_attack_today": float(p_given_attack),
        "p_attack_tomorrow_given_no_attack_today": float(p_given_none),
        "marginal_attack_rate": float(b.mean()),
        "risk_ratio": (float(p_given_attack / p_given_none)
                       if p_given_none else float("nan")),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "n_pairs": int(len(a)),
    }


def per_patient_transition(series_by_patient, min_pairs: int = 20):
    """Per-patient conditional risks, for patients with enough observed
    consecutive pairs and at least one attack-today day. Feeds the
    heterogeneity view (some patients cluster, some do not).
    """
    rows = []
    for pid, s in series_by_patient.items():
        v = s.to_numpy(dtype=float)
        if len(v) < 2:
            continue
        x0, x1 = v[:-1], v[1:]
        ok = ~np.isnan(x0) & ~np.isnan(x1)
        x0, x1 = x0[ok], x1[ok]
        if len(x0) < min_pairs or (x0 == 1).sum() == 0:
            continue
        rows.append({
            "patient_id": pid,
            "p_given_attack": float(x1[x0 == 1].mean()),
            "p_given_none": float(x1[x0 == 0].mean()) if (x0 == 0).any() else float("nan"),
            "n_pairs": int(len(x0)),
            "n_attack_days": int((x0 == 1).sum()),
        })
    return rows
