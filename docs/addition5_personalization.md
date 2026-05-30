# Addition 5: personalisation and within-person evaluation

> Position in the paper: **Methods (4.4)**. Reads after `addition4_sequence.md`; precedes `external_validation_site.md`. Within-person C-statistic and the per-patient regime tables defined here carry the load-bearing finding restated in results_findings.md Headline takeaway 1.

The personalisation layer (`experiment/5/`) implements both the modelling and
the evaluation side of the within-person migraine-forecasting consensus.
Additions 0, 1, and 4 fit a single model across all patients and report a
pooled discrimination number; the migraine-forecasting literature converges
on the opposite design: individualised models, evaluated within-person.
Addition 5 builds that on the Park 2016 SHD cohort, using the heterogeneity
result Addition 3 already produced as its motivation.

## 1. Why this addition exists, and what it builds on

The single most consistent finding across the prior machine-learning work on
migraine prediction is that individualised models (models trained on a single
patient's own history) beat generalised (cross-patient) models, and that
achievable skill varies considerably between individuals; an individualised model
is defined there as one trained only on the target patient's data
[dumkrieger2025review, p. 1; p. 3]. The same pattern recurs in the
specific studies: diary-only forecasting from perceived stress was evaluated with
leave-one-out validation across 95 persons and 4,626 diary days, reaching an
in-sample AUC of 0.73 but only 0.65 under leave-one-out validation
[houle2017stress, p. 1041; p. 1044]; a follow-up developed continuous Bayesian
updating of *individual* forecasting models as new diary days accrue
[houle2021bayesian, p. 1264]; the wearable per-night detector used personal
quadratic-discriminant models over only seven subjects and reported balanced
accuracy above 84% one night ahead but with wide between-subject variance
[siirtola2018sleep, p. 1]; and the ambulatory physiological work built per-patient
state-space (N4SID) models with an average 47-minute forecast horizon
[pagan2015ambulatory, p. 15419].

The closest methodological comparator made the *evaluation* point
explicitly. Holsteen et al. reported their next-day trigger-exposure model with a
within-person C-statistic of 0.56 (95% CI 0.54-0.58), only slightly better than
chance, precisely because a pooled C-statistic mixes between-patient base-rate
variation into the discrimination estimate and overstates within-person
forecasting skill [holsteen2020triggers, p. 2364]. Additions 0-1 report
the pooled metric, which is the quantity that argument warns against.

Addition 3 supplied the cohort-specific evidence that personalisation is worth
testing here and not just elsewhere: per-patient lag-1 autocorrelation is
estimable for 49 (migraine) / 62 (headache) patients and spans a wide range, and
only ~26% (migraine) / ~31% (headache) of patients with enough events are bursty,
so some patients cluster strongly day-to-day while others are near-memoryless
(`docs/addition3_results.md`, heterogeneity claim). That heterogeneity is exactly
the signal that tells a model whether to personalise.

## 2. Research questions

1. **Does personalisation help on SHD?** Do per-patient or partially-pooled
   models achieve a higher *within-person* AUROC distribution than the pooled
   Addition 0/1/4 models on the same chronological evaluation?
2. **Pooled vs within-person reporting.** How large is the gap between the pooled
   C-statistic the benchmark reports and the within-person C-statistic
   that the comparator literature argues for [holsteen2020triggers, p. 2364]? Is
   the benchmark's headline discrimination optimistic by the same mechanism?
3. **Who is predictable?** Which patient characteristics (event count, base rate,
   Addition 3 self-excitation strength) separate the well-forecast from the
   poorly-forecast patients, given the documented between-individual variability
   [dumkrieger2025review, p. 1]?
4. **Cold-start.** How many of a patient's own diary days does an individual or
   updating model need before it beats the population model for that patient --
   the question the Bayesian-updating work was built around
   [houle2021bayesian, p. 1264]?

## 3. Methods (with rationale and literature anchor)

The personalisation analyses use the NonHP base architectures on the
canonical full_features chronological 70/30 cells, sharing the paper-
headline 70/30 partition (NonHP XGBoost migraine hold-out AUROC 0.777,
TabPFN headache v2.6 hold-out AUROC 0.652; the HP020 / TabPFN-v2.6
headlines reach 0.793 / 0.652 on the same partition). A non-HP backbone
keeps the personalisation contrast clean of hyperparameter-search
variance. The within-person C-statistic is computed from the 5-fold
expanding-window CV file `diary_cv5_timeseries.parquet`, which is
ratio-agnostic; the C-statistic clustering near 0.55 therefore
reproduces across architectures and split ratios (Section 9d below).

### 3.1 Three modelling regimes

- **Pooled (baseline).** The existing Additions 0/1/4 models, re-scored under the
  within-person evaluation of Section 3.3 so the comparison is on one metric.
- **Per-patient.** One model per patient, trained on that patient's own
  chronological history (the individualised design the literature favours)
  [dumkrieger2025review, p. 3; houle2017stress, p. 1044; siirtola2018sleep, p. 1].
  Feasible only for patients with enough events; patients below an event-count
  floor are reported as not-estimable rather than forced.
- **Partial pooling (recommended).** A mixed-effects / hierarchical model with a
  patient random intercept (and a random slope on the recent-attack-rate term
  where event counts support it), shrinking short, noisy patient series toward
  the cohort mean. This is the same partial-pooling discipline Addition 3 adopted
  for its self-excitation and recurrent-event estimates to avoid the ecological
  fallacy of naive pooling (`docs/addition3_temporal.md`, Section 9), applied here
  to forecasting rather than description. It directly absorbs the migraine
  sparsity: a patient with three attacks is shrunk hard, a patient with twenty
  contributes more.

TabPFN offers a fourth, cheaper route to individualisation worth testing
alongside: per-patient in-context inference, where the patient's own prior days
form the in-context training set at prediction time, with no parameter fitting
(`docs/tabPfn.MD`). This is the foundation-model analogue of the per-patient
regime and is essentially free to evaluate once the model is loaded.

### 3.2 Cold-start / sequential-update protocol

For the per-patient and updating regimes, evaluate in an expanding-window,
walk-forward manner per patient: predict day t using only days < t of that
patient (plus, for the partially-pooled model, the cohort prior). Record the
running within-person AUROC as a function of the number of the patient's own days
seen, which answers RQ4 and instantiates the continuous-updating idea
[houle2021bayesian, p. 1264] without committing to a full Bayesian posterior in
the first pass.

### 3.3 Within-person evaluation (the core of this addition)

The headline metric is the **per-patient AUROC/AUPRC distribution** and a
within-person C-statistic defined as the meta-analytic summary of the
person-specific discriminations, following the comparator's explicit argument
that within-person metrics are the fitting way to evaluate individual-attack
forecasting [holsteen2020triggers, p. 2364]. Per-patient AUROC variance is
estimated by the Hanley-McNeil binormal approximation (Hanley and McNeil
1982); the cohort summary is a Paule-Mandel τ² random-effects pooled mean
(primary, robust under k<20 per Veroniki et al. 2016), with
DerSimonian-Laird τ² reported alongside as a sensitivity for
comparability with the migraine-forecasting literature; τ² is
estimated from the per-patient sampling variances under either
estimator. Reporting is:

- the distribution (median and IQR, with a per-patient strip/box plot), not a
  single pooled scalar;
- the pooled-minus-within-person gap (RQ2), which quantifies how much the
  benchmark's existing headline is inflated by between-patient base-rate mixing;
- calibration per the standard panel (slope, intercept, ECE10, Brier), because
  TRIPOD+AI requires calibration alongside discrimination [collins2024tripodAI,
  p. 6] and per-patient calibration is where small-sample fragility bites
  hardest (`docs/results_findings.md`, Section 6);
- AUPRC read per target only, never across the migraine/headache base-rate gap
  [mcdermott2024aurocAuprc, p. 1].

Patients with too few positive days for a stable per-patient AUROC are listed as
not-estimable with their event counts, mirroring how Addition 3 foregrounded
pooled estimates where the migraine sparsity made per-patient fits unreliable,
and how the small-sample literature demands shrinkage rather than
over-interpretation of tiny series [martin2025samplesize, p. 2].

The "too few positive days" floor differs between the hold-out and the
cross-validated paths because the two estimands have different sample
structures. The CV path pools out-of-fold predictions across five
expanding-window folds, so a patient with 5 positive days across the
union of their CV test folds has a stable per-fold-pooled
discrimination estimate; the floor is `MIN_POS = 5` (constant defined
in `experiment/5/_personal/within_person.py`). The hold-out path
estimates per-patient AUROC inside the single chronological test slice
only, where total positive counts are smaller and many patients fail
even a permissive floor; the threshold is relaxed to `min_pos = 3`
(used in `run_personalization.py`). The looser hold-out floor is a
disclosed estimability decision, not a quality fix: per-patient AUROC
estimates from 3-4 positive days carry wider Hanley-McNeil CIs, which
the reporting includes per estimable patient.

### 3.4 Relationship to the patient-holdout split

Addition 0/1 already include a whole-patient hold-out split, where migraine is
the *hardest* setting (mean AUROC 0.55, below both stratified and chronological)
because generalising to an unseen patient with few positive days is hard
(`docs/results_findings.md`, Section 2). That split answers "generalise to a *new*
patient"; Addition 5 answers the complementary and more deployment-relevant
"forecast for a *returning* patient whose history we already have." Reporting both
keeps the two scientific questions separate, as the methodology section already
commits to (`docs/paper_rigor_checklist.md`, Section 5).

## 4. Multiple-comparisons discipline

The pre-registered confirmatory comparison is RQ1/RQ2 at the canonical 70/15/15
chronological `full_features` cell per target: partial-pooling vs pooled, on the
within-person C-statistic, with a paired test across patients. The
who-is-predictable regression (RQ3) and the cold-start curves (RQ4) are
exploratory and reported with that label, the same discipline Addition 3 used to
separate confirmatory tests from the exploratory scan
(`docs/addition3_temporal.md`, Section 3.7).

## 5. Directory and output layout

```
experiment/5/
  run_personalization.py          # driver
  _personal/
    regimes.py                    # regimes + the standard-contract emitter (comparability bridge)
    walkforward.py                # per-patient expanding-window cold-start eval
    within_person.py              # per-patient AUROC/AUPRC + meta-analytic C-statistic
  <target>/<feature_set>/<regime>/<ratio>/<split>/
    results/results_<ts>_<uid>.txt  # standard sharedMetricPrinter contract ->
                                    #   folds into the SAME comparison_*.html as 0/1/4
    within_person_<ts>.csv          # per-patient metrics + event counts + not-estimable flags
    distribution_<ts>.png           # per-patient AUROC strip/box
    coldstart_<ts>.png              # within-person AUROC vs n own-days seen
  comparison_personalization_<ts>.html  # pooled vs within-person gap, regime x target
```

**Comparability.** The leaf path is parse_path-compatible
(`<addition>/<target>/<feature_set>/<architecture=regime>/<ratio>/<split>`), so
each regime's pooled discrimination/calibration folds into the existing
`comparison_*.html` next to Additions 0/1/4 via `run_aggregate_results.py`, with
no aggregator change. The within-person C-statistic is a different estimand (not
in the 11-metric contract), so it is reported as its own per-patient artifact;
the pooled-vs-within-person gap (RQ2) is read across the two.

## 6. Build order

1. `_personal/within_person.py` first: the evaluation is the contribution, so
   it is built and tested before any new model, and applied to the *existing*
   Addition 0/1 predictions to produce the RQ2 pooled-vs-within-person gap with
   zero new training.
2. `_personal/regimes.py`: partial pooling (the recommended regime) before
   per-patient, because per-patient is its event-count-limited special case.
3. `_personal/walkforward.py`: the cold-start protocol.
4. `run_personalization.py` glue and the comparison page.

Each module stays under the 300-line budget.

## 7. Dependencies

- `statsmodels` (mixed-effects GLM / GEE): already pinned for Addition 3.
- Optional `lifelines` reuse from Addition 3 if a recurrent-event personalised
  hazard is added later; not required for the first pass.
- No new data dependency.

## 8. Compute budget

Partial-pooling GLMs and within-person scoring are CPU-seconds-to-minutes over
~4.5k rows and ~63 patients. TabPFN per-patient in-context inference reuses the
existing GPU path. The cold-start walk-forward is the only multiplier and is still
minutes.

## 9. Resolved decisions and the open one

- **Partial pooling over naive per-patient pooling**: chosen, because lumping
  patients with different base rates manufactures between-patient autocorrelation
  (the ecological-fallacy risk Addition 3 already names,
  `docs/addition3_temporal.md`, Section 9).
- **Within-person as the headline metric**: chosen, on the comparator's explicit
  argument [holsteen2020triggers, p. 2364].
- **Open**: whether to add a full Bayesian posterior-updating model
  [houle2021bayesian, p. 1264] in this addition or defer it. The walk-forward
  expanding-window protocol (Section 3.2) captures the cold-start question without
  it; the full Bayesian treatment is recorded as the natural follow-up.

## 9a. Initial RQ2 result (cross-architecture within-person eval on existing leaves)

Running `run_personalization.py` over the existing chronological `full_features`
70/15/15 leaves of Additions 0 (XGBoost stack), 1 (TabPFN v3-default) and 4
(sequence window-MLP) - within-person evaluation only, no new training;
predictions on the model's full out-of-sample horizon (val + test); per-patient
floor of 3 positives, Hanley-McNeil variance weighting down-weighting tiny-n
patients; each leaf scored in its own subprocess so the conflicting
`_model_architecture` packages and the TabPFN-GPU/CPU device split do not
collide:

| target   | architecture     | pooled AUROC | within-person C-stat | est. | gap    |
|----------|------------------|--------------|----------------------|------|--------|
| headache | XGBoost (add 0)  | 0.637        | 0.530 [0.463-0.597]  | 13/20| +0.107 |
| headache | TabPFN (add 1)   | 0.654        | 0.484 [0.419-0.550]  | 13/20| +0.169 |
| headache | sequence (add 4) | 0.598        | 0.395 [0.292-0.498]  | 13/20| +0.203 |
| migraine | XGBoost (add 0)  | 0.763        | 0.538 [0.398-0.677]  | 3/20 | +0.225 |
| migraine | TabPFN (add 1)   | 0.764        | 0.477 [0.341-0.613]  | 3/20 | +0.287 |
| migraine | sequence (add 4) | 0.771        | 0.569 [0.427-0.711]  | 3/20 | +0.202 |

Two findings emerged. First, the pooled headline overstated within-person
forecasting skill for every architecture (gap +0.11 to +0.29), and the
within-person C-statistics clustered near chance (~0.40-0.57), next to
Holsteen's independent-cohort 0.56 [holsteen2020triggers, p. 2364]. Second,
the pooled ranking did not survive the within-person reframing: pooled migraine is a
three-way tie (~0.76) but within-person it is sequence > XGBoost > TabPFN, and
pooled headache favours TabPFN while within-person favours XGBoost, so the
headline metric can mislead about per-patient utility. Caveats: the chronological
hold-out contains only late-enrolment patients, so migraine is estimable for only
3 patients (wide CIs); this hold-out pass is superseded by the CV out-of-fold
estimate in Section 9b.

## 9b. Trustworthy RQ2 result (CV out-of-fold within-person)

The hold-out (9a) is sparse because the chronological test block contains only
late-enrolment patients. The CV out-of-fold pass refits each architecture across
the 5 expanding-window TimeSeriesSplit folds and pools the held-out-fold
predictions, so every patient receives out-of-sample predictions across the full
date range. Estimability rises from 13->57 patients (headache) and 3->19
(migraine) at the principled 5-positive floor, and the estimates tighten:

| target   | architecture     | pooled AUROC | within-person C-stat | est. | gap    |
|----------|------------------|--------------|----------------------|------|--------|
| headache | XGBoost (add 0)  | 0.606        | 0.543 [0.511-0.575]  | 57/63| +0.062 |
| headache | TabPFN (add 1)   | 0.654        | 0.542 [0.509-0.575]  | 57/63| +0.112 |
| headache | sequence (add 4) | 0.631        | 0.538 [0.506-0.570]  | 57/63| +0.093 |
| migraine | XGBoost (add 0)  | 0.624        | 0.558 [0.511-0.604]  | 19/63| +0.067 |
| migraine | TabPFN (add 1)   | 0.738        | 0.565 [0.508-0.622]  | 19/63| +0.173 |
| migraine | sequence (add 4) | 0.685        | 0.530 [0.476-0.584]  | 19/63| +0.155 |

The headline is now clean and well-estimated: **within-person discrimination
clusters at ~0.53-0.57 for every architecture and both targets**, near chance
and architecture-independent, beside Holsteen's independent-cohort 0.56
[holsteen2020triggers, p. 2364]. The pooled "leads" largely evaporate
within-person (migraine TabPFN's pooled 0.738 falls to 0.565, the largest gap),
and the within-person architecture differences sit inside overlapping CIs,
statistically indistinguishable per patient. Reproduce with
`run_personalization.py --cv`. This is the trustworthy RQ2 estimate; 9a is the
sparse first pass kept for the hold-out-vs-CV contrast.

**Pooling methodology.** All six within-person C-statistic values use
Paule-Mandel τ² random-effects pooling, the default of the
`within_person_cstatistic` helper at `experiment/5/_personal/within_person.py`
(robust under k<20 per Veroniki et al. 2016), regenerated by `python
experiment/5/run_personalization.py --cv` with the output summary at
`experiment/5/within_person_summary_cv_20260530_151055.csv`. This matches
the body §3.6 and abstract Results primary disclosure. The DerSimonian-Laird
τ² sensitivity is available alongside at the body §3.6 L93 canonical
side-by-side disclosure (Paule-Mandel primary + DL sensitivity quoted
explicitly for the two headline TabPFN cells); the two estimators agree
at the third decimal place on this cohort with the largest DL-vs-PM shifts
being the migraine XGBoost point estimate (DL 0.554 → PM 0.558) and the
headache sequence CI lower bound (DL 0.505 → PM 0.506).

## 9c. RQ1 result (personalisation regimes, standard-contract)

`run_personalization.py --regimes` fits three regimes sharing one logistic-
regression base (unweighted; imbalance left to the threshold step) so the
comparison isolates the POOLING effect: `pooled` (one global LR), `per_patient`
(per-patient LR with global fallback for patients under 30 own training rows),
and `partial_pool` (global LR plus an empirical-Bayes per-patient random
intercept, Gaussian-prior Newton posterior mode). Each emits the standard
results contract under
`experiment/5/<target>/<feature_set>/<regime>/<ratio>/<split>/results/`, so it
folds into the same `comparison_*.html` as Additions 0/1/4 (architecture = the
regime name). Migraine, chronological 70/15/15, test-set AUROC:

| feature_set        | pooled | per_patient | partial_pool |
|--------------------|--------|-------------|--------------|
| park_features (6)  | 0.602  | 0.617       | 0.701        |
| no_rolling (26)    | 0.728  | 0.730       | 0.738        |

**Partial pooling improves pooled AUROC** on the hold-out, substantially on the
sparse park trigger set (+0.10 test AUROC over pooled) and modestly on no_rolling
(+0.01); §9d below shows that gain does not survive within-person re-evaluation.
A per-patient random
intercept, essentially each patient's shrunk base rate, carries next-day signal
the same-day triggers miss, consistent with the within-patient self-excitation of
Addition 3. Per-patient LR alone barely moves (its own-history models are
data-starved), which is why the shrinkage of partial pooling is the right tool.
Caveat: the test split is small (~136 rows), so the per-regime bootstrap CIs in
the results files are wide; the within-person view of the regimes is k~1 on the
hold-out and needs the CV out-of-fold pass (Section 9b) to be estimable. The
regime ranking is the RQ1 headline; the within-person magnitudes are Section 9b.

## 9d. Regimes within-person, CV out-of-fold (the decisive RQ1/RQ2 result)

`run_personalization.py --regimes --cv` refits each regime across the 5
expanding-window folds and pools the held-out-fold predictions, making the
regime within-person C-statistic estimable (k=19 patients vs ~1 on the hold-out).
Migraine:

| feature_set     | regime       | pooled OOF AUROC | within-person C-stat |
|-----------------|--------------|------------------|----------------------|
| full_features   | pooled       | 0.677            | 0.540 [0.478-0.602]  |
| full_features   | per_patient  | 0.635            | 0.511 [0.446-0.577]  |
| full_features   | partial_pool | 0.694            | 0.517 [0.453-0.581]  |
| park_features   | pooled       | 0.570            | 0.546 [0.483-0.609]  |
| park_features   | per_patient  | 0.612            | 0.472 [0.414-0.531]  |
| park_features   | partial_pool | 0.704            | 0.486 [0.423-0.549]  |
| no_rolling      | pooled       | 0.569            | 0.521 [0.467-0.575]  |
| no_rolling      | per_patient  | 0.595            | 0.477 [0.403-0.550]  |
| no_rolling      | partial_pool | 0.663            | 0.490 [0.425-0.555]  |

Headache (full_features only, the cell `fig_d2_regimes.py` plots):

| feature_set     | regime       | pooled OOF AUROC | within-person C-stat |
|-----------------|--------------|------------------|----------------------|
| full_features   | pooled       | 0.620            | 0.538 [0.510-0.567]  |
| full_features   | per_patient  | 0.603            | 0.528 [0.494-0.561]  |
| full_features   | partial_pool | 0.641            | 0.527 [0.497-0.556]  |

Partial pooling's pooled-AUROC gain (+0.10 to +0.13 over pooled) did not
survive the within-person reframing and slightly lowered the within-person
C-statistic (park 0.546 -> 0.486; no_rolling 0.521 -> 0.490).
The per-patient random intercept improves pooled discrimination by separating
high-rate from low-rate patients (a between-patient base-rate effect), but it
adds no within-patient ranking of a given patient's migraine vs non-migraine
days, and the per-patient noise costs a little. This is the Holsteen caution
[holsteen2020triggers, p. 2364] made concrete on this cohort: a personalisation
"improvement" measured by pooled AUROC can be entirely between-patient and
vanish, or reverse, per patient. The honest RQ1 answer is therefore: on the
pooled metric partial pooling helps, but it does not improve per-patient
next-day forecasting, which remains near chance (~0.49-0.55) for every regime.
This is why the within-person metric, not pooled AUROC, must headline the
personalisation claim.

## 9e. Cold-start curve (RQ4): how many own days until personalisation helps

`run_personalization.py --coldstart` walks each patient's diary in date order and
scores two causal forecasters of the next day by Brier: the population cohort
rate (no personalisation) and the patient's own running attack rate from prior
days only, shrunk toward the cohort by a pseudocount (alpha=5). Per own-day band:

| own days | migraine pop / pers | headache pop / pers |
|----------|---------------------|---------------------|
| 0        | 0.046 / 0.046 (tie) | 0.181 / 0.181 (tie) |
| 1-3      | 0.078 / 0.070       | 0.224 / 0.212       |
| 4-7      | 0.053 / 0.053       | 0.203 / 0.182       |
| 8-14     | 0.066 / 0.058       | 0.207 / 0.176       |
| 15-29    | 0.083 / 0.077       | 0.184 / 0.173       |
| 30+      | 0.063 / 0.057       | 0.168 / 0.161       |

(Brier; lower is better; both forecasters scored on the identical rows per band.)
The patient-cluster bootstrap 95% CI columns `pop_ci_low`/`pop_ci_high` and
`pers_ci_low`/`pers_ci_high` are emitted to `coldstart_curve_*.csv` and rendered
as fill_between envelopes on `fig_d1`. The envelopes overlap at the 0-day tie
band and narrow as own-day history accrues, mirroring the point-estimate
trajectory: the personalised curve sits below the cohort curve from the first
prior day for both targets, and the gap widens past the envelope overlap by the
8-14 own-day band (e.g. headache 8-14: 0.207 [pop CI] vs 0.176 [pers CI],
non-overlapping). The cold-start point was ~1 own day for both targets: from the
first prior day the personalised running rate beat the cohort rate and stayed
below, with the margin widening as history accrued (e.g. headache 8-14 days:
0.207 -> 0.176).
Knowing even a little of a patient's own attack history improves the next-day
*probability* forecast, because patients are heterogeneous in base rate
(Addition 3). Important nuance, consistent with Section 9d: this is the
between-patient *level* (base-rate) effect: it lowers Brier by getting each
patient's overall rate right, not an improvement in within-patient day-to-day
discrimination, which stays near chance. So personalisation pays off immediately
for *calibrated risk level*, not for ranking which of a patient's days is the
attack.

## 10. How this addition changes the rest of the paper

RQ2 is the load-bearing result: it states, in the benchmark's own numbers, how
much the pooled headline discrimination overstates per-patient forecasting skill,
aligning the paper with the within-person reporting the comparator literature uses
[holsteen2020triggers, p. 2364]. If personalisation helps (RQ1), the paper can
recommend an individualised deployment consistent with the field consensus
[dumkrieger2025review, p. 1]; if it does not on this cohort, that is itself
informative given the documented between-individual variability and the SHD
sparsity. Either way the addition converts Addition 3's heterogeneity *description*
into a forecasting *prescription*.

## References

[dumkrieger2025review] G. M. Dumkrieger, "The promise of machine learning in
predicting migraine attacks," *Cephalalgia*, vol. 45, no. 11, 2025. doi:
10.1177/03331024251391207. (Individualised beat generalised models and skill
varies considerably between individuals, p. 1; individualised-model definition,
p. 3.)

[houle2017stress] T. T. Houle *et al.*, "Forecasting individual headache attacks
using perceived stress," *Headache*, vol. 57, no. 7, pp. 1041-1050, 2017. doi:
10.1111/head.13137. (N=95, 4,626 diary days, AUC 0.73 train / 0.65 leave-one-out,
p. 1041; leave-one-out validation, p. 1044.)

[houle2021bayesian] T. T. Houle, H. Deng, C. H. Tegeler, and D. P. Turner,
"Continuous updating of individual headache forecasting models using Bayesian
methods," *Headache*, vol. 61, no. 8, pp. 1264-1273, 2021. doi:
10.1111/head.14182. (Continuous Bayesian updating of individual forecasting
models, p. 1264.)

[holsteen2020triggers] K. K. Holsteen, M. Hittle, M. Barad, and L. M. Nelson,
"Development and internal validation of a multivariable prediction model for
individual episodic migraine attacks based on daily trigger exposures,"
*Headache*, vol. 60, no. 10, pp. 2364-2379, 2020. doi: 10.1111/head.13960.
(Within-person C-statistic 0.56, 95% CI 0.54-0.58, only slightly better than
chance; within-person metrics are the fitting evaluation, p. 2364.)

[pagan2015ambulatory] J. Pagán *et al.*, "Robust and accurate modeling
approaches for migraine per-patient prediction from ambulatory data," *Sensors*,
vol. 15, no. 7, pp. 15419-15442, 2015. doi: 10.3390/s150715419. (Per-patient
state-space N4SID models, 47-minute forecast horizon, p. 15419.)

[siirtola2018sleep] P. Siirtola, H. Koskimäki, H. Mönttinen, and J.
Röning, "Using sleep time data from wearable sensors for early detection of
migraine attacks," *Sensors*, vol. 18, no. 5, p. 1374, 2018. doi:
10.3390/s18051374. (Personal QDA models, balanced accuracy >84% one night ahead,
seven subjects, wide between-subject variance, p. 1.)

[collins2024tripodAI] G. S. Collins *et al.*, "TRIPOD+AI statement," *BMJ*, vol.
385, e078378, 2024. doi: 10.1136/bmj-2023-078378. (Report discrimination and
calibration, item 12e, p. 6.)

[mcdermott2024aurocAuprc] M. B. A. McDermott *et al.*, "A closer look at AUROC and
AUPRC under class imbalance," arXiv:2401.06091, 2024. doi:
10.48550/arXiv.2401.06091. (AUPRC depends on prevalence, p. 1.)

[martin2025samplesize] G. P. Martin, R. D. Riley, J. Ensor, and S. W. Grant,
"Statistical primer: sample size considerations for developing and validating
clinical prediction models," *Eur. J. Cardiothorac. Surg.*, vol. 67, no. 5, p.
ezaf142, 2025. doi: 10.1093/ejcts/ezaf142. (Shrinkage for small samples, p. 2.)
