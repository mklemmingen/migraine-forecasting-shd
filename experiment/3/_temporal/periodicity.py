"""Day-of-week periodicity of the attack series.

Attack rate by day of week with a chi-square goodness-of-fit test against a
constant rate: the expected attacks per weekday are proportional to the
number of observed (non-missing) days on that weekday, so uneven weekday
sampling does not by itself create apparent structure. Literature anchor:
the chronobiology systematic review reports a Saturday weekly peak
[poulsen2021chronobiology]. The diary is day-resolution, so the weekly axis
is tested and the circadian axis is out of scope.
"""
import numpy as np
from scipy.stats import chisquare

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def day_of_week_counts(series_by_patient):
    """Pooled observed-day and attack counts per weekday over the recorded
    (non-missing) calendar days across patients.
    """
    obs = np.zeros(7)
    att = np.zeros(7)
    for s in series_by_patient.values():
        v = s.to_numpy(dtype=float)
        dow = np.asarray(s.index.dayofweek)
        ok = ~np.isnan(v)
        for d in range(7):
            mask = ok & (dow == d)
            obs[d] += int(mask.sum())
            att[d] += float(np.nansum(v[mask]))
    return obs, att


def periodicity_test(series_by_patient) -> dict:
    """Chi-square goodness-of-fit of weekday attack counts against a constant
    rate, with the peak and trough weekday.
    """
    obs, att = day_of_week_counts(series_by_patient)
    total_att, total_obs = att.sum(), obs.sum()
    if total_obs == 0 or total_att == 0:
        return {"n_attacks": int(total_att)}
    expected = total_att * (obs / total_obs)
    chi2, p = chisquare(att, f_exp=expected)
    rate = np.divide(att, obs, out=np.zeros(7), where=obs > 0)
    peak, trough = int(np.argmax(rate)), int(np.argmin(rate))
    return {
        "obs_days_per_dow": obs.tolist(),
        "attacks_per_dow": att.tolist(),
        "rate_per_dow": [round(float(r), 4) for r in rate],
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": 6,
        "peak_weekday": WEEKDAYS[peak],
        "peak_rate": float(rate[peak]),
        "trough_weekday": WEEKDAYS[trough],
        "trough_rate": float(rate[trough]),
        "n_attacks": int(total_att),
    }
