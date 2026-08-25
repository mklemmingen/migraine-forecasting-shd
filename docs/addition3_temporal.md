# Addition 3: temporal-dependence analysis of the SHD cohort

> Position in the long-form record: **Methods**. Reads after `addition2_explainability.md`; precedes `addition4_sequence.md`. The temporal-dependence verdict from this doc gates the sequence-modelling decision documented in addition4_sequence.md (the IID-row falsification test).

The temporal-dependence layer (`experiment/3/`) trained no models. It
read the engineered diary parquets directly and characterised the
temporal structure of the attack series. The purpose
is to test, on the SHD cohort, whether next-day attacks depend on the
recent attack history, and to feed that answer into the decision of
whether a sequence model (Addition 4) is justified over the tabular
approaches of Additions 0 and 1.

## 1. Why this addition exists, and what it builds on

Additions 0 and 1 treat each diary day as an independent row. That IID
assumption is convenient but unverified. If migraine days are strongly
self-exciting or autocorrelated, then (a) the IID framing understates
the achievable forecast skill, (b) random and stratified splits leak
temporal information across train and test, and (c) a sequence model
that consumes the attack history is the methodologically correct next
step. If instead the series is close to memoryless given the same-day
triggers, the tabular framing is defensible and Addition 4 would be
hard to justify. Addition 3 produces the evidence that settles this,
rather than assuming either way.

Addition 3 builds a calendar-regular gap-aware series of the daily
attack indicator per patient, then runs a formal serial-dependence test,
a burstiness decomposition, a history-conditioned regression that
controls for same-day triggers, a higher-order transition analysis, and
a day-of-week periodicity test. The data pipeline's
`dataset_analysis.py::_page_temporal_dependence` carries a related but
calendar-naive ACF (computed on the compacted record index, so a lag of
k means k records, not k calendar days); the calendar-regular reindexer
in `experiment/3/_temporal/series.py` operates on calendar days rather than record indices. The output is a
standalone report that ends in an explicit Addition-4 verdict.

The migraine literature already reports the expected direction.
Headache days cluster, with day 1 a good predictor of day 2
[houle2005timeseries, p. 445]; attack onset shows weekly (Saturday-peak) and
circadian periodicity [poulsen2021chronobiology, p. 1]; and Markov-chain
models of attack counts have been published [barra2020markov, p. 3].
Addition 3 therefore tests pre-specified hypotheses on the SHD cohort
rather than scanning for significance.

## 2. Research questions

1. **Serial dependence**: is the daily attack series autocorrelated
   beyond what the same-day trigger features explain? At what lags
   (1 day, last week, last month)?
2. **Self-excitation / clustering**: do attacks arrive in bursts (an
   attack raises the near-future attack rate), as a regular cycle, or
   as a memoryless Poisson stream?
3. **History-conditioned hazard**: does the time since the last attack,
   or the attack count in the trailing window, change the probability
   of an attack tomorrow once same-day triggers are controlled?
4. **Periodicity**: is there a day-of-week or within-cohort weekly
   structure consistent with the chronobiology literature?

## 3. Methods (with rationale and literature anchor)

Six complementary analyses, each answering one facet. They are
deliberately a mix of descriptive, point-process, and
regression-based tools so a single method's assumptions do not carry
the conclusion.

### 3.1 Autocorrelation function and Ljung-Box test

Per-patient and pooled autocorrelation function (ACF) of the binary
daily attack indicator, with Ljung-Box portmanteau tests for serial
dependence at lags up to 30 days. `statsmodels` is already pinned for
exactly this. Literature anchor: Houle et al. 2005 found day-1 to day-2
positive autocorrelation [houle2005timeseries, p. 445]. Gap handling is the
subtlety: the diary has 208 transitions with gaps > 1 day (max 37). The
ACF is computed on the calendar-regular reindexed series with explicit
missing days rather than on the compacted record index, so a lag of k
means k calendar days, not k records.

### 3.2 Burstiness and memory coefficients

The Goh-Barabasi burstiness parameter B and memory coefficient M
computed on inter-attack gap times [goh2008burstiness, p. 2]. B in (-1, 1)
separates regular (B < 0), Poisson (B near 0), and bursty (B > 0)
series; M separates whether short gaps tend to follow short gaps. Goh
and Barabasi found human event streams are bursty mainly through the
inter-event distribution rather than memory, so reporting B and M
separately is the established decomposition. These are scalar summaries
per patient, easy to plot as a cohort distribution.

### 3.3 Discrete-time self-excitation (not continuous Hawkes)

A continuous-time self-exciting (Hawkes-type) point process was
considered and rejected for this data. Such a process is a
continuous-time model whose likelihood assumes precise event
timestamps with continuous inter-event densities. The SHD diary is
day-resolution binary: inter-event times are integers and attacks
cannot be ordered within a day, which violates the continuous-density
assumption the standard self-exciting-process MLE rests on. Forcing
day-indexed integers into the continuous likelihood would be a misuse,
so the continuous formulation was not used. (The originating Hawkes
1971 reference is not cited because the source PDF could not be
obtained; the rejection rests on the data-resolution argument, not on
that paper's content, so no citation is needed for the choice.)

The scientifically correct discrete analog is a discrete-time
self-exciting model: an autoregressive logistic regression

    P(attack_t = 1 | history) = sigmoid( b0 + sum_k b_k * attack_{t-k}
                                          + same-day trigger terms )

fit on the calendar-regular reindexed series. Self-excitation appears
as positive coefficients b_k on the recent attack lags, estimated
*after* controlling for the same-day trigger features, so it isolates
history dependence beyond what today's triggers already explain. This
is correct for the data resolution, dovetails with the recurrent-event
regression in 3.4 (both ask whether history predicts the next attack
beyond same-day triggers), and is reported with the lag coefficients,
their CIs, and a likelihood-ratio test against the triggers-only model.

### 3.4 History-conditioned recurrent-event regression

Andersen-Gill counting-process regression and the
Prentice-Williams-Peterson gap-time variant [amorim2015recurrent, p. 326]
on inter-attack gap times, with the trailing
attack rate and days-since-last-attack as covariates. AG estimates the
overall effect on the event intensity; PWP-gap-time estimates the
effect conditional on the number of prior events, which is the
forecasting-relevant quantity. This connects the temporal analysis
directly to the prediction framing: if days-since-last-attack carries a
significant hazard ratio after adjusting for same-day triggers, the
forecasting models are leaving signal on the table by treating rows as
independent.

### 3.5 First-order Markov transition matrix

The simplest answer to "does the past predict the future":
P(attack tomorrow | attack today) versus
P(attack tomorrow | no attack today), per patient and pooled, with a
chi-square test of independence and a comparison to the marginal attack
rate. Literature anchor: Markov-chain attack modelling
[barra2020markov, p. 3]. This is the headline number a clinical reader
will understand without point-process background, so it leads the
report even though the Hawkes and recurrent-event models are more
complete.

### 3.6 Day-of-week and weekly periodicity

Attack rate by day of week with a chi-square goodness-of-fit against
uniform, and a circular-statistics test for a preferred weekday.
Literature anchor: the chronobiology systematic review reports a
Saturday weekly peak and morning circadian peak
[poulsen2021chronobiology, p. 1]; the SHD diary is day-resolution so we test
the weekly axis and note the circadian axis as out of scope for
day-level data.

### 3.7 Multiple-comparisons discipline

The six analyses across two targets, per-patient and pooled, are a
large test battery (ACF to lag 30 alone is 30 correlated tests per
series). Reporting raw p-values across all of them would inflate the
family-wise error rate and is a PROBAST+AI analysis-domain risk. We
therefore pre-register a small set of confirmatory primary tests - the
pooled lag-1 ACF, the pooled first-order Markov transition, the pooled
self-excitation likelihood-ratio test against the triggers-only model,
and the day-of-week omnibus - one per target. Everything else (every
higher lag, every per-patient fit, the secondary periodicity tests) is
labelled exploratory. Within each analysis family a BH-FDR correction is applied via a
Benjamini-Hochberg false-discovery-rate correction and report adjusted
p-values, so the burstiness, recurrent-event, and per-lag results are
read as a controlled-FDR set rather than a scan for significance. The
"hypothesis testing, not fishing" framing of Section 1 is the design
intent; this subsection is the statistical mechanism that enforces it.

## 4. Two targets reported separately

The analysis runs on both the any-headache series (~23.5% of days) and
the ICHD-3 migraine series (~7.2% of days). The migraine series is
sparse: per-patient attack counts are low, so per-patient Hawkes and
recurrent-event fits are underpowered and we foreground the pooled
estimates with explicit uncertainty. The headache series is denser and
carries the per-patient analyses better. Reporting both, and being
explicit about which estimates the migraine sparsity makes unreliable,
is the rigour line here.

## 5. Directory and output layout

```
experiment/3/
  run_temporal_analysis.py        # driver: load parquets, run the six analyses, emit report
  _temporal/
    series.py                     # gap-aware calendar-regular event series (reuses analytics logic)
    acf.py                        # ACF + Ljung-Box (3.1), extends analytics ACF
    burstiness.py                 # Goh-Barabasi B, M (3.2)
    self_excitation.py            # discrete-time autoregressive-logistic self-excitation (3.3)
    recurrent.py                  # Andersen-Gill / PWP-gap-time (3.4)
    markov.py                     # first-order transition matrix + chi-square (3.5)
    periodicity.py                # day-of-week chi-square + circular test (3.6)
  <target>/
    acf_pooled_<ts>.png
    acf_perpatient_<ts>.png       # small-multiples grid
    burstiness_scatter_<ts>.png   # B vs M cohort scatter
    self_excitation_<ts>.png      # lag coefficients with CIs
    markov_transition_<ts>.png
    dow_periodicity_<ts>.png
    temporal_report_<ts>.html     # all figures + a stats table with tests, p-values, CIs
  temporal_summary_<ts>.html      # one-page cross-target verdict feeding the Addition-4 decision
```

The `temporal_summary` page states the Addition-4 implication
explicitly: "self-excitation or autocorrelation present, sequence model
justified" or "series memoryless given same-day triggers, tabular
framing defensible". That single verdict is the deliverable Addition 4
depends on.

## 6. Build order

1. `_temporal/series.py`: the gap-aware calendar-regular event series
   per patient, factored out of the existing analytics page so both
   the PDF report and this addition share one implementation. Built and
   unit-tested first (a synthetic patient with a known 37-day gap
   verifies the lag arithmetic); every other module consumes its
   output.
2. `_temporal/markov.py` and `_temporal/acf.py`: the two
   lowest-assumption analyses, which also sanity-check the series
   construction (a positive lag-1 ACF should match the Markov
   off-diagonal). `acf.py` adds the Ljung-Box test the analytics page
   lacks.
3. `_temporal/burstiness.py`, `_temporal/periodicity.py`: scalar
   summaries, cheap and additive.
4. `_temporal/self_excitation.py`, `_temporal/recurrent.py`: the
   regression analyses, which carry the trigger-controlled history
   coefficients; built last so the simpler results frame them.
5. `run_temporal_analysis.py`: glue + the cross-target summary verdict.

## 7. Dependencies

- `lifelines` (Andersen-Gill / PWP recurrent-event regression; mature,
  pure-Python, no compiled point-process stack required)
- The discrete-time self-excitation model is logistic regression with
  lagged features; the partial-pooling version adds a patient random
  intercept via `statsmodels` mixed-effects GLM (or GEE for a
  population-averaged variant), so it needs only `statsmodels`, already
  present. No `tick`, no continuous-time point-process dependency
  (Section 3.3).
- `statsmodels` (ACF, Ljung-Box) and `seaborn` (small-multiples) are
  already pinned; `scipy.stats` (chi-square, circular tests) ships with
  scipy.

## 8. Compute budget

All analyses are CPU-only over ~4.5k diary rows across ~63 patients.
Total run was under five minutes on a single CPU core. No GPU, no contention with the
sweep.

## 9. Estimation strategy: partial pooling

Each patient is their own time series; dependence is estimated within
patient and combined with a random-effects model that shrinks short,
noisy series toward the cohort mean. Naive pooling of raw patient-days
manufactures autocorrelation that lives between patients, not within any
patient (ecological-fallacy / Simpson's-paradox risk). The
self-excitation and recurrent-event models carry a patient random
intercept (and a random slope on the history term where per-patient
event counts support it); the ACF, burstiness, and Markov summaries are
reported as per-patient distributions with a precision-weighted cohort
estimate overlaid. The continuous-time Hawkes route is excluded: the
day-resolution data makes a continuous point process inappropriate, so
the discrete-time autoregressive-logistic model is the methodologically
correct analogue.

## 9a. Research insights this enables for paper claims

The chosen design yields three distinct, defensible claims that
neither pooling extreme alone supports:

1. **Cohort-level headline** (random-effects pooled estimate): "Across
   the SHD cohort, an attack today raises the probability of an attack
   tomorrow by X percentage points after adjusting for same-day
   triggers (95% CI ...)." This is the citable effect size, with
   uncertainty that honestly reflects between-patient variation rather
   than the false precision of pooled patient-days.
2. **Heterogeneity claim** (per-patient distribution): "Self-excitation
   is not universal: N of the cohort's patients show strong day-to-day
   clustering, M show none." This is the clinically richer statement,
   and it is the one that tells a downstream sequence model (Addition
   4) whether it should personalise (large heterogeneity) or fit a
   shared dynamic (small heterogeneity).
3. **Methods-integrity claim** (cross-addition consequence): a
   significantly positive within-patient autocorrelation explains why
   random/stratified splits were optimistic; the mechanism was the
   FEATURE CHANNEL, not the model. patient_id is not a feature (dropped
   before fitting) and the tabular models are row-order-invariant, so
   nothing leaks through learned order; what leaks is the history
   feature VALUES (migraine_yesterday, *_rate_last7, streaks), which
   encode a patient's adjacent days that random shuffling scatters
   across the train/test boundary. This is verified empirically in
   `docs/results_findings.md` Section 1: the stratified-minus-chrono
   AUROC inflation appears only for feature sets that contain history
   features (full +0.03..+0.11, spano +0.03..+0.09) and is absent for
   no_rolling and park (~-0.03). So chronological splitting is the
   honest evaluation specifically when history features are present;
   for the no-history feature sets the stratified-vs-chrono gap is
   interpolation-vs-extrapolation, not leakage. This sharpens the
   patient_id-leakage caveat documented for hyperparameter tuning into
   a feature-channel statement.

These three are the paper payload of Addition 3: an effect size, a
heterogeneity result, and a methods-integrity result that retroactively
strengthens the evaluation design of Additions 0 and 1.

## 10. How this addition changes the rest of the paper

This is the one addition whose result can retroactively constrain the
others. If it finds strong self-excitation, the methods section must
add a paragraph acknowledging that the random and stratified splits in
Additions 0 and 1 leak temporal structure (the chronological split
results become the trustworthy ones, consistent with the
patient_id-leakage finding already documented for hyperparameter
tuning). If it finds weak dependence, it becomes the explicit
justification for not pursuing a sequence model, turning the open
question of whether a sequence model is warranted on this cohort into
a cited, tested answer.

## References

[houle2005timeseries] T. T. Houle *et al.*, "Time-series features of
headache: individual distributions, patterns, and predictability of
pain," *Headache*, vol. 45, no. 5, pp. 445-458, 2005.

[goh2008burstiness] K.-I. Goh and A.-L. Barabasi, "Burstiness and
memory in complex systems," *EPL*, vol. 81, no. 4, p. 48002, 2008.

[amorim2015recurrent] L. D. A. F. Amorim and J. Cai, "Modelling
recurrent events: a tutorial for analysis in epidemiology," *Int. J.
Epidemiol.*, vol. 44, no. 1, pp. 324-333, 2015,
doi:10.1093/ije/dyu222. AG and PWP-GT both defined on p. 326.

[poulsen2021chronobiology] A. H. Poulsen, M. Younis, T. Thuraiaiyah,
and M. Ashina, "The chronobiology of migraine: a systematic review,"
*J. Headache Pain*, vol. 22, p. 76, 2021.

[barra2020markov] M. Barra, F. A. Dahl, K. Vetvik, and E. A.
MacGregor, "A Markov chain method for counting and modelling migraine
attacks," *Scientific Reports*, vol. 10, p. 3631, 2020.

[park2016shd] J.-W. Park *et al.*, "Analysis of trigger factors in
episodic migraineurs using a smartphone headache diary applications,"
*PLOS ONE*, vol. 11, no. 2, p. e0149577, 2016.
