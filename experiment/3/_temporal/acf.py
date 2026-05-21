"""Autocorrelation of the daily attack series, calendar-correct.

For each lag k the pooled autocorrelation is the Pearson correlation of the
(x_t, x_{t+k}) pairs that do not straddle a missing calendar day, pooled
across patients (the series module supplies the gap-respecting pairs), so a
lag of k is always k calendar days rather than k records. Per-lag 95% bands
are +/-1.96/sqrt(n_pairs_k).

The standard Ljung-Box portmanteau assumes a gap-free regular series; the
SHD diary is irregular, so the portmanteau below is computed on the pooled
pairwise autocorrelations with an effective sample size and is reported as
approximate. The exact, pre-registered lag-1 serial-dependence test is the
first-order Markov chi-square in markov.py (for a binary series the lag-1
autocorrelation and the 2x2 transition association are equivalent).
Literature anchor: Houle et al. 2005 found day-1 to day-2 positive
autocorrelation [houle2005timeseries].
"""
import numpy as np
from scipy.stats import chi2

import series as S


def pooled_acf(series_by_patient, nlags: int = 30) -> dict:
    """Pooled calendar-correct ACF with per-lag pair counts and 95% bands.

    Returns ``{lag: {"r": float, "n_pairs": int, "ci": float}}`` for lags
    0..nlags. Lag 0 is 1.0 by definition.
    """
    out = {0: {"r": 1.0, "n_pairs": 0, "ci": 0.0}}
    for lag in range(1, nlags + 1):
        a, b = S.pooled_indicator_pairs(series_by_patient, lag)
        if len(a) < 2 or a.std() == 0 or b.std() == 0:
            out[lag] = {"r": float("nan"), "n_pairs": int(len(a)), "ci": float("nan")}
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        ci = float(1.96 / np.sqrt(len(a)))
        out[lag] = {"r": r, "n_pairs": int(len(a)), "ci": ci}
    return out


def approx_ljung_box(acf_table: dict, h: int = 14) -> dict:
    """Approximate Ljung-Box portmanteau over lags 1..h.

    Q = n_eff (n_eff + 2) sum_k r_k^2 / (n_eff - k), with the effective
    sample size taken as the lag-1 pooled pair count. Reported as approximate
    because the diary's calendar gaps break the regular-series assumption the
    exact statistic rests on; the lag-1 Markov chi-square is the exact
    primary test.
    """
    n_eff = acf_table.get(1, {}).get("n_pairs", 0)
    if n_eff <= h + 1:
        return {"Q": float("nan"), "p_value": float("nan"), "h": h, "n_eff": int(n_eff)}
    q = 0.0
    for k in range(1, h + 1):
        r = acf_table.get(k, {}).get("r", float("nan"))
        if np.isnan(r):
            continue
        q += (r * r) / (n_eff - k)
    Q = n_eff * (n_eff + 2) * q
    p = float(chi2.sf(Q, df=h))
    return {"Q": float(Q), "p_value": p, "h": h, "n_eff": int(n_eff)}


def per_patient_acf1(series_by_patient, min_pairs: int = 20) -> list:
    """Per-patient lag-1 autocorrelation, for the heterogeneity view.

    Only patients with enough observed consecutive pairs and a non-constant
    attack series contribute.
    """
    rows = []
    for pid, s in series_by_patient.items():
        v = s.to_numpy(dtype=float)
        if len(v) < 2:
            continue
        x0, x1 = v[:-1], v[1:]
        ok = ~np.isnan(x0) & ~np.isnan(x1)
        x0, x1 = x0[ok], x1[ok]
        if len(x0) < min_pairs or x0.std() == 0 or x1.std() == 0:
            continue
        rows.append({"patient_id": pid, "r1": float(np.corrcoef(x0, x1)[0, 1]),
                     "n_pairs": int(len(x0))})
    return rows
