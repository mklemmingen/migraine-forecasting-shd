"""Goh-Barabasi burstiness and memory of the attack series.

On each patient's inter-attack gap times (calendar days between consecutive
attack days), the burstiness parameter B = (sd - mean) / (sd + mean) in
[-1, 1] separates regular (B < 0), Poisson (B near 0), and bursty (B > 0)
streams; the memory coefficient M is the lag-1 correlation of consecutive
gaps. Goh and Barabasi 2008 found human event streams are bursty mainly
through the inter-event-time distribution rather than memory, so B and M
are reported separately [goh2008burstiness].

A recording gap can lengthen an apparent inter-attack time (an unrecorded
attack could fall inside the gap), so estimates are reported per patient
with the event count and as a cohort distribution rather than as a single
pooled scalar.
"""
import numpy as np


def _attack_dates(series):
    """Observed attack calendar days (index positions where value == 1)."""
    return series.index[series.to_numpy(dtype=float) == 1.0]


def patient_burstiness(series_by_patient, min_attacks: int = 4) -> list:
    """Per-patient B and M, for patients with at least ``min_attacks`` attack
    days (B needs a stable gap distribution; M needs at least three gaps).
    """
    rows = []
    for pid, s in series_by_patient.items():
        dates = _attack_dates(s)
        if len(dates) < min_attacks:
            continue
        tau = np.diff(np.asarray(dates, dtype="datetime64[D]")).astype(float)
        if len(tau) < 2:
            continue
        mu = float(tau.mean())
        sd = float(tau.std(ddof=0))
        B = (sd - mu) / (sd + mu) if (sd + mu) > 0 else 0.0
        if len(tau) >= 3 and tau[:-1].std() > 0 and tau[1:].std() > 0:
            M = float(np.corrcoef(tau[:-1], tau[1:])[0, 1])
        else:
            M = float("nan")
        rows.append({
            "patient_id": pid,
            "n_attacks": int(len(dates)),
            "n_gaps": int(len(tau)),
            "mean_gap_days": mu,
            "B": float(B),
            "M": M,
        })
    return rows


def cohort_summary(rows: list) -> dict:
    """Cohort distribution of B and M with the share of bursty patients."""
    B = np.array([r["B"] for r in rows], dtype=float)
    M = np.array([r["M"] for r in rows if not np.isnan(r["M"])], dtype=float)
    if B.size == 0:
        return {"n_patients": 0}
    return {
        "n_patients": int(B.size),
        "B_median": float(np.median(B)),
        "B_iqr": [float(np.percentile(B, 25)), float(np.percentile(B, 75))],
        "M_median": float(np.median(M)) if M.size else float("nan"),
        "frac_bursty_B_positive": float(np.mean(B > 0)),
        "frac_regular_B_negative": float(np.mean(B < 0)),
    }
