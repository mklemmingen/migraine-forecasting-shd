"""Figures for the temporal-dependence report.

Reuses the journal figure style (vector PDF + 300-DPI PNG, embedded fonts)
shared with the benchmark figures, so Addition 3 is visually consistent
with the rest of the paper. Each function takes a results dict from one
analysis module and writes one figure.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiment"))
from _style import apply, save  # noqa: E402

# Okabe-Ito colour-blind-safe accents.
_BLUE = "#0072B2"
_ORANGE = "#E69F00"
_GREEN = "#009E73"
_GREY = "#999999"


def acf_plot(acf_table, target, out_path):
    """ACF stems over lags 1..n with the +/-95% band shaded."""
    apply()
    lags = [k for k in sorted(acf_table) if k >= 1]
    r = [acf_table[k]["r"] for k in lags]
    ci = [acf_table[k]["ci"] for k in lags]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(lags, [-c for c in ci], ci, color=_GREY, alpha=0.25,
                    label="95% band")
    ax.vlines(lags, 0, r, color=_BLUE, lw=2)
    ax.plot(lags, r, "o", color=_BLUE, ms=4)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("Lag (calendar days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Daily attack autocorrelation - {target}")
    ax.legend(fontsize=8)
    save(fig, out_path)
    plt.close(fig)


def burstiness_plot(rows, target, out_path):
    """Per-patient burstiness B vs memory M scatter with zero quadrants."""
    apply()
    B = [x["B"] for x in rows]
    M = [x["M"] for x in rows]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.axhline(0, color=_GREY, lw=0.8)
    ax.axvline(0, color=_GREY, lw=0.8)
    ax.scatter(B, M, color=_ORANGE, edgecolor="#333", s=30, alpha=0.85)
    ax.set_xlabel("Burstiness B  (-1 regular, 0 Poisson, +1 bursty)")
    ax.set_ylabel("Memory M (lag-1 gap correlation)")
    ax.set_title(f"Inter-attack burstiness and memory - {target}")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    save(fig, out_path)
    plt.close(fig)


def dow_plot(period, target, out_path):
    """Attack rate by weekday."""
    apply()
    from periodicity import WEEKDAYS
    rate = period["rate_per_dow"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(WEEKDAYS, rate, color=_GREEN, edgecolor="#333")
    ax.set_ylabel("Attack rate")
    ax.set_title(f"Attack rate by day of week - {target} "
                 f"(chi-square p = {period['p_value']:.2f})")
    save(fig, out_path)
    plt.close(fig)


def self_excitation_plot(se, target, out_path):
    """Forest plot of attack-lag odds ratios (trigger-controlled)."""
    apply()
    coefs = se["lag_coefs"]
    names = list(coefs)
    or_ = [coefs[n]["odds_ratio"] for n in names]
    lo = [np.exp(coefs[n]["ci"][0]) for n in names]
    hi = [np.exp(coefs[n]["ci"][1]) for n in names]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.errorbar(or_, y, xerr=[np.subtract(or_, lo), np.subtract(hi, or_)],
                fmt="o", color=_BLUE, capsize=3)
    ax.axvline(1.0, color=_GREY, lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Odds ratio (trigger-controlled)")
    ax.set_title(f"Attack-lag self-excitation - {target}\n"
                 f"LR vs triggers-only: chi2={se['lr_stat']:.0f}, "
                 f"p={se['lr_p_value']:.1e}")
    save(fig, out_path)
    plt.close(fig)


def markov_plot(mk, target, out_path):
    """Conditional next-day attack probabilities vs the marginal rate."""
    apply()
    labels = ["no attack today", "attack today", "marginal"]
    vals = [mk["p_attack_tomorrow_given_no_attack_today"],
            mk["p_attack_tomorrow_given_attack_today"],
            mk["marginal_attack_rate"]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, vals, color=[_GREY, _ORANGE, _BLUE], edgecolor="#333")
    ax.set_ylabel("P(attack tomorrow)")
    ax.set_title(f"First-order transition - {target}  "
                 f"(RR={mk['risk_ratio']:.1f}, p={mk['p_value']:.1e})")
    save(fig, out_path)
    plt.close(fig)
