# Addition 3 results: temporal dependence of the SHD attack series

> Position in the paper: **Results (5.2)**. Reads after `addition2_results.md`; precedes `insights_leaf_selection.md`. Per-Addition temporal-dependence findings; references the methods specified in addition3_temporal.md.


Data-verified findings from `experiment/3/run_temporal_analysis.py` on the
unsplit diaries (`data/processed/<target>/diary.parquet`), 63 patients,
calendar-regular gap-aware daily attack series (208 gap transitions, max 37
days, 9.55% of calendar days missing). Six analyses on both targets; the
plan and rationale live in `docs/addition3_temporal.md`. Headline: the
daily attack series carries strong short-range within-patient serial
dependence in both targets, confirmed by four independent methods.

## Research questions (Section 2 of the plan)

### RQ1 serial dependence: YES, strong, beyond same-day triggers
- Pooled ACF (calendar-correct, gap-aware): migraine lag-1 r = 0.267,
  decaying over a week (0.167, 0.137, 0.108, ...); headache lag-1 r = 0.218.
- Discrete-time self-excitation (trigger-controlled logistic, lags 1-3 +
  same-day stress/sleep/weather/overeating): migraine lag-1 OR 5.43
  (p = 3e-21), lag-2 2.08, lag-3 2.13; likelihood-ratio vs triggers-only
  chi2 = 155, p = 2.5e-33. Headache lag-1 OR 2.44; LR chi2 = 203,
  p = 1.1e-43. Attack history therefore predicted the next day after
  adjustment for same-day triggers. Consistent with Houle et al. 2005's
  day-1 -> day-2 positive autocorrelation [houle2005timeseries, p. 445].

### RQ2 clustering vs cycle vs Poisson: short-range clustering, not bursts
- First-order Markov: migraine P(attack tomorrow | attack today) = 0.315 vs
  0.050 without (risk ratio 6.3, chi2 = 298, p = 8.8e-67); headache 0.397
  vs 0.179 (RR 2.2, p = 1.8e-45) [barra2020markov, p. 3].
- Goh-Barabasi burstiness: median B = -0.05 (migraine) / -0.06 (headache),
  i.e. the inter-attack-TIME distribution is near-Poisson, not heavy-tailed
  bursty; memory M near zero to slightly negative [goh2008burstiness, p. 2]. The
  dependence is therefore short-range day-to-day clustering (captured by the
  lag-1 ACF / Markov / self-excitation), not classical burstiness in the
  inter-event times. Reporting B and M separately is the established
  decomposition and it sharpens the claim: "clustered" here means
  short-memory autocorrelation, not a long-tailed bursty stream.

### RQ3 history-conditioned hazard: YES
- A higher trailing 14-day attack rate shortened the time to the next
  attack beyond the marginal rate. Recurrent-event regression on the
  inter-attack gap times, with Andersen-Gill counting-process estimation
  and the Prentice-Williams-Peterson gap-time variant
  [amorim2015recurrent, p. 326], yielded migraine Andersen-Gill HR 3.50
  (p = 0.001), PWP-gap-time HR 3.71 (p = 3e-05); headache AG 2.79, PWP
  3.36 (p = 5e-21).

### RQ4 periodicity: no significant weekly structure
- Day-of-week chi-square goodness-of-fit: migraine chi2 = 5.0, p = 0.54
  (numerical peak Saturday, trough Monday); headache chi2 = 5.2, p = 0.52
  (peak Friday). The migraine Saturday peak matches the direction reported
  by the chronobiology review [poulsen2021chronobiology, p. 1] but is not
  significant in this cohort; the circadian axis is out of scope for
  day-resolution data.

## Multiple-comparisons discipline (Section 3.7)

Three confirmatory tests were pre-registered per target: the pooled
lag-1 Markov transition, the self-excitation likelihood-ratio test, and
the day-of-week omnibus, all reported above with raw p-values. The
exploratory ACF lags 2-14 are Benjamini-Hochberg FDR-corrected in the
per-target HTML report. The serial-dependence conclusion rests on the
confirmatory set, not on a scan over lags.

## Addition-4 verdict

**A sequence model is justified.** Both targets showed significant
within-patient self-excitation beyond same-day triggers (migraine LR
p = 2.5e-33, headache p = 1.1e-43). The dependence was short-range (lag-1
dominant, decaying by ~day 3-7) rather than long-range bursty, so a model
consuming the recent attack history (a few days) captures the available
signal; a long-memory architecture is not indicated by the burstiness
result. Per-patient heterogeneity (below) suggests personalisation is worth
testing alongside a shared dynamic.

## Three paper claims (Section 9a)

1. **Cohort effect size.** In migraine, an attack today raised the
   probability of an attack tomorrow from 5.0% to 31.5% (risk ratio 6.3);
   the trigger-controlled lag-1 odds ratio was 5.4. In headache, the
   probability rose from 17.9% to 39.7% (RR 2.2). These were the citable
   effect sizes.
2. **Heterogeneity.** Self-excitation is not universal: per-patient lag-1
   autocorrelation is estimable for 49 (migraine) / 62 (headache) patients
   and spans a wide range, and only ~26% (migraine) / 31% (headache) of
   patients with enough events are bursty (B > 0). Some patients cluster
   strongly day-to-day; others are near-memoryless. This is the result that
   tells a sequence model whether to personalise.
3. **Methods-integrity (cross-addition).** The strong within-patient serial
   dependence is the mechanism behind the optimistic bias of random and
   stratified splits documented in `docs/results_findings.md` Section 1. The
   tabular models are row-order-invariant, so nothing leaks through learned
   order; what leaks is the history FEATURE values (migraine_yesterday,
   *_rate_last7, streaks) that encode a patient's adjacent days, which
   random shuffling scatters across the train/test boundary. This is why
   the stratified-minus-chronological AUROC inflation appears only for
   feature sets that contain history features (full, spano) and is absent
   for no_rolling and park. Addition 3 supplies the temporal evidence that
   the leakage caveat is real and quantifies the dependence it rests on.

## Caveats

Migraine attacks are sparse (~7% of recorded days), so per-patient
estimates are noisy and the pooled estimates are foregrounded; recording
gaps can lengthen an apparent inter-attack time, which is why the burstiness
and recurrent-event analyses are reported with their event counts. The
day-level resolution precludes any circadian (within-day) analysis. The
approximate Ljung-Box statistic is reported as approximate because calendar
gaps break the regular-series assumption; the exact lag-1 test is the Markov
chi-square.

## Provenance of the headline numbers

Every numerical value cited above is persisted as machine-readable JSON
next to the per-target HTML report:
`experiment/3/migraine/headline_<timestamp>.json` and
`experiment/3/headache/headline_<timestamp>.json`. The JSON includes
the pooled lag-1 ACF (r and pair count), Markov contingency
(P(attack|attack today), risk ratio, chi-square, p), the
self-excitation LR statistic with degrees of freedom and per-lag odds
ratios, Goh-Barabasi burstiness median B and M with the bursty
fraction, and Andersen-Gill and PWP-gap-time hazard ratios with their
Cox p-values. Cross-validation against this JSON should reproduce
every number in the section above to three decimal places.

