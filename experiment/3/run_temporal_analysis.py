"""Driver for the temporal-dependence analysis (Addition 3).

Runs the six analyses on both targets, applies the pre-registered primary
test / BH-FDR discipline (docs/addition3_temporal.md Section 3.7), writes
per-target figures and an HTML report, and a cross-target summary stating
the Addition-4 verdict (does the serial dependence justify a sequence
model). Reads the unsplit diary parquets only; fits no forecasting models.
"""
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "_temporal"))
from _eval._html_to_pdf import html_to_pdf  # noqa: E402
import series as S          # noqa: E402
import markov as MK         # noqa: E402
import acf as ACF           # noqa: E402
import burstiness as BU     # noqa: E402
import periodicity as PD    # noqa: E402
import self_excitation as SE  # noqa: E402
import recurrent as RC      # noqa: E402
import _plots as PL         # noqa: E402

TARGETS = ("migraine", "headache")


def _acf_pvalue(r, n):
    """Two-sided normal-approximation p-value for a sample autocorrelation."""
    if n <= 3 or np.isnan(r):
        return float("nan")
    from scipy.stats import norm
    return float(2.0 * norm.sf(abs(r) * np.sqrt(n)))


def run_target(target):
    df = S.load_diary(target)
    col = S.attack_column(df)
    sbp = S.build_patient_series(df, col)
    acf_table = ACF.pooled_acf(sbp, nlags=30)
    burst_rows = BU.patient_burstiness(sbp)
    res = {
        "gap": S.gap_summary(df),
        "markov": MK.pooled_transition(sbp),
        "markov_pp": MK.per_patient_transition(sbp),
        "acf": acf_table,
        "acf_lb": ACF.approx_ljung_box(acf_table, h=14),
        "acf_pp": ACF.per_patient_acf1(sbp),
        "burst_rows": burst_rows,
        "burst": BU.cohort_summary(burst_rows),
        "period": PD.periodicity_test(sbp),
        "se": SE.fit_self_excitation(target),
        "recurrent": RC.fit_recurrent(target),
    }
    return sbp, res


def fdr_table(res):
    """Pre-registered primary tests (reported raw) plus a BH-FDR-corrected
    exploratory family (the higher ACF lags 2..14)."""
    primary = {
        "lag-1 Markov transition": res["markov"]["p_value"],
        "self-excitation LR (vs triggers-only)": res["se"].get("lr_p_value", float("nan")),
        "day-of-week omnibus": res["period"]["p_value"],
    }
    expl_lags = list(range(2, 15))
    raw = [_acf_pvalue(res["acf"][k]["r"], res["acf"][k]["n_pairs"]) for k in expl_lags]
    finite = [(k, p) for k, p in zip(expl_lags, raw) if not np.isnan(p)]
    if finite:
        _rej, adj, _a, _b = multipletests([p for _k, p in finite], method="fdr_bh")
        exploratory = {f"ACF lag {k}": (p, a) for (k, p), a in zip(finite, adj)}
    else:
        exploratory = {}
    return primary, exploratory


def headline_scalars(target, res):
    """Flat dict of the numbers cited in docs/addition3_results.md.

    Persisting these alongside the HTML report closes the verifiability
    gap: every number the Results doc claims (pooled lag-1 ACF r, Markov
    chi2, self-excitation LR chi2 and per-lag ORs, burstiness B and M,
    Andersen-Gill and PWP-gap-time HRs) is recoverable from this JSON.
    """
    acf, mk, se, bu, rc = res["acf"], res["markov"], res["se"], res["burst"], res["recurrent"]
    out = {
        "target": target,
        "pooled_acf_lag1_r": acf[1]["r"],
        "pooled_acf_lag1_n_pairs": acf[1]["n_pairs"],
        "markov_chi2": mk.get("chi2", float("nan")),
        "markov_p_value": mk.get("p_value", float("nan")),
        "markov_p_attack_given_attack": mk.get("p_attack_tomorrow_given_attack_today", float("nan")),
        "markov_p_attack_given_none": mk.get("p_attack_tomorrow_given_no_attack_today", float("nan")),
        "markov_risk_ratio": mk.get("risk_ratio", float("nan")),
        "selfexc_lr_chi2": se.get("lr_stat", float("nan")),
        "selfexc_lr_df": se.get("lr_df", float("nan")),
        "selfexc_lr_p_value": se.get("lr_p_value", float("nan")),
        "selfexc_lag1_OR": se.get("lag_coefs", {}).get("lag1", {}).get("odds_ratio", float("nan")),
        "selfexc_lag2_OR": se.get("lag_coefs", {}).get("lag2", {}).get("odds_ratio", float("nan")),
        "selfexc_lag3_OR": se.get("lag_coefs", {}).get("lag3", {}).get("odds_ratio", float("nan")),
        "burstiness_B_median": bu.get("B_median", float("nan")),
        "burstiness_M_median": bu.get("M_median", float("nan")),
        "burstiness_frac_bursty": bu.get("frac_bursty_B_positive", float("nan")),
        "burstiness_n_patients": bu.get("n_patients", 0),
        "AG_HR_prior_rate": rc.get("ag", {}).get("hr_prior_rate", float("nan")),
        "AG_p_value": rc.get("ag", {}).get("p_value", float("nan")),
        "PWP_HR_prior_rate": rc.get("pwp", {}).get("hr_prior_rate", float("nan")),
        "PWP_p_value": rc.get("pwp", {}).get("p_value", float("nan")),
        "n_recurrent_intervals": rc.get("n_intervals", 0),
        "n_recurrent_events": rc.get("n_events", 0),
    }
    return out


def write_figures(target, res, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    names = {}
    PL.acf_plot(res["acf"], target, out_dir / f"acf_{ts}.png"); names["acf"] = f"acf_{ts}.png"
    PL.burstiness_plot(res["burst_rows"], target, out_dir / f"burstiness_{ts}.png"); names["burst"] = f"burstiness_{ts}.png"
    PL.dow_plot(res["period"], target, out_dir / f"dow_{ts}.png"); names["dow"] = f"dow_{ts}.png"
    PL.self_excitation_plot(res["se"], target, out_dir / f"self_excitation_{ts}.png"); names["se"] = f"self_excitation_{ts}.png"
    PL.markov_plot(res["markov"], target, out_dir / f"markov_{ts}.png"); names["markov"] = f"markov_{ts}.png"
    return names, ts


def report_html(target, res, fig_names, primary, exploratory, headline):
    mk, se, pr = res["markov"], res["se"], res["period"]
    rows = "".join(
        f"<tr><td>{k}</td><td>{p:.2e}</td></tr>" for k, p in primary.items())
    expl = "".join(
        f"<tr><td>{k}</td><td>{p:.2e}</td><td>{a:.2e}</td></tr>"
        for k, (p, a) in exploratory.items())
    imgs = "".join(f'<img src="{n}" style="max-width:680px;display:block;margin:8px 0">'
                   for n in fig_names.values())
    headline_rows = "".join(
        f"<tr><td>{k}</td><td>{v if isinstance(v, int) else f'{v:.4g}' if isinstance(v, float) else v}</td></tr>"
        for k, v in headline.items() if k != "target")
    return f"""<!doctype html><meta charset=utf-8>
<title>Temporal dependence - {target}</title>
<body style="font-family:sans-serif;max-width:760px;margin:24px auto">
<h2>Temporal dependence - {target}</h2>
<p>Patients {res['burst']['n_patients']}; gap transitions
{res['gap']['n_gap_transitions']} (max {res['gap']['max_gap_days']} days);
missing-day fraction {res['gap']['missing_day_fraction']}.</p>
<p>Markov: P(attack tomorrow | attack today) = {mk['p_attack_tomorrow_given_attack_today']:.3f}
vs {mk['p_attack_tomorrow_given_no_attack_today']:.3f} without (risk ratio
{mk['risk_ratio']:.1f}). Self-excitation lag-1 OR
{se['lag_coefs']['lag1']['odds_ratio']:.2f} (trigger-controlled).</p>
<h3>Headline scalars (machine-readable: headline_*.json)</h3>
<table border=1 cellpadding=4 style="border-collapse:collapse"><tr><th>field</th><th>value</th></tr>{headline_rows}</table>
<h3>Pre-registered primary tests (raw p)</h3>
<table border=1 cellpadding=4 style="border-collapse:collapse"><tr><th>test</th><th>p</th></tr>{rows}</table>
<h3>Exploratory ACF lags (BH-FDR adjusted)</h3>
<table border=1 cellpadding=4 style="border-collapse:collapse"><tr><th>lag</th><th>p</th><th>p_adj</th></tr>{expl}</table>
<h3>Figures</h3>{imgs}
</body>"""


def verdict(all_res):
    """Addition-4 implication from the primary self-excitation tests."""
    lines = []
    justified = True
    for t in TARGETS:
        se = all_res[t]["se"]
        sig = se.get("lr_p_value", 1.0) < 0.05
        justified = justified and sig
        lines.append(
            f"{t}: self-excitation LR p={se.get('lr_p_value', float('nan')):.1e}, "
            f"lag-1 OR {se['lag_coefs']['lag1']['odds_ratio']:.2f}; "
            f"Markov RR {all_res[t]['markov']['risk_ratio']:.1f}.")
    head = ("Sequence model JUSTIFIED: significant within-patient "
            "self-excitation beyond same-day triggers in both targets."
            if justified else
            "Sequence model NOT clearly justified: weak serial dependence.")
    return head, lines


def main():
    all_res = {}
    fig_idx = {}
    for t in TARGETS:
        _sbp, res = run_target(t)
        all_res[t] = res
        primary, exploratory = fdr_table(res)
        names, ts = write_figures(t, res, HERE / t)
        headline = headline_scalars(t, res)
        headline_path = HERE / t / f"headline_{ts}.json"
        headline_path.write_text(json.dumps(headline, indent=2, sort_keys=True))
        report_path = HERE / t / f"temporal_report_{ts}.html"
        report_path.write_text(report_html(t, res, names, primary, exploratory, headline))
        html_to_pdf(report_path)
        fig_idx[t] = ts
        print(f"{t}: report written ({len(names)} figures); headline JSON {headline_path.name}")
    head, lines = verdict(all_res)
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    summary = ("<!doctype html><meta charset=utf-8><title>Temporal summary</title>"
               "<body style='font-family:sans-serif;max-width:760px;margin:24px auto'>"
               f"<h2>Temporal-dependence summary</h2><p><b>{head}</b></p><ul>"
               + "".join(f"<li>{x}</li>" for x in lines) + "</ul></body>")
    summary_path = HERE / f"temporal_summary_{ts}.html"
    summary_path.write_text(summary)
    html_to_pdf(summary_path)
    print("VERDICT:", head)


if __name__ == "__main__":
    main()
