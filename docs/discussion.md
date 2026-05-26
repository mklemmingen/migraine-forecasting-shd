# Discussion

> Position in the paper: **Discussion (7)**. Reads after the Results section (`results_findings.md` and the per-Addition results docs); precedes the Conclusion (`conclusion.md`). Synthesises the cross-Addition implications, positions the result against the literature, and bounds the claim.

## 7.1 The pooled-AUROC framing overstates within-person value on this cohort

The benchmark's load-bearing finding is that the pooled AUROC commonly
reported in the next-day-diary migraine-forecasting literature
overstates within-person forecasting skill on the Park 2016 SHD cohort.
On the chronological headline cell the migraine AUROC reaches 0.793
(*XGB-HP020 / full / chrono / 70-30*) and the headache AUROC 0.652
(TabPFN family at *full / chrono / 70-30*: v2.6, v3-default, and
v3-binary tied within the composite rule's 0.02 AUROC tier, v2.6 picked
by calibration-distance tiebreak). The per-patient AUROC distribution
on the same leaves centres near 0.55, the precision-weighted
within-person C-statistic clusters near the same value, and the Brier
skill against each patient's own TRAIN-set climatology is negative for
migraine (-0.05 to -0.12) while positive for headache (+0.14 to +0.21).
Pooled discrimination on this cohort is therefore overwhelmingly
between-patient base-rate separation, not within-person day-to-day
ranking. The implication for clinical deployment is direct: a 24-hour
migraine forecast that beats chance under pooled evaluation does not
necessarily let a given patient act on *their* day-by-day predictions.

This finding reproduces the within-person Holsteen 2020 pattern on an
independent cohort. Holsteen et al. reported a within-person C-statistic
of 0.56 (95% CI 0.54-0.58) on a 178-patient US trigger-exposure cohort
and argued that the pooled C-statistic mixes between-patient base-rate
variation into the discrimination estimate, so within-person metrics are
the fitting evaluation for an individualised forecast
[holsteen2020triggers, p. 2364]. Our within-person C-statistic on the
Park cohort sits in the same near-chance band; the benchmark therefore
adds a second cohort's evidence to the Holsteen argument and extends it
with a per-patient AUROC distribution, a Brier skill, and a
decision-curve net benefit that none of the previous diary-only
forecasting papers reported simultaneously.

## 7.2 Each Addition supplies an independent line of evidence

The pooled-overstates conclusion does not rest on any single
measurement. Five Additions converge on it through different
methodological lenses.

Addition 0 establishes that the depth-diverse XGBoost stack reaches its
migraine headline AUROC of 0.793 only on the HP-tuned 70_30 chronological
cell and that the same architecture's calibration slope is fragile across
the 488-cell grid (median 0.64, 98 cells with negative slope, exhibiting
the Platt-inversion fingerprint of small-calibration-set overfitting).
Addition 1 shows that the TabPFN family matches or slightly exceeds
XGBoost on the headache headline cell without per-leaf hyperparameter
tuning, and is markedly more robust than HP-tuned XGBoost on small or
leakage-prone splits. Read together, Additions 0 and 1 demonstrate that
the architecture choice is the smallest source of variation in this
benchmark; the splits and the metric definition matter more.

Addition 2 layers SHAP attribution on the calibrated probability and
finds that the history features (`migraine_rate_last7`,
`headache_free_streak`, `days_since_last_migraine`, `migraine_yesterday`)
carry 77-85% of the mean absolute attribution on the full_features cells,
while the same model on the no_rolling feature set (which contains no
history columns) puts 0% attribution on history. The history-feature
share is the same for chronological and stratified splits, but the
neighbour-averaging features specifically gain rank and magnitude under
stratification: `migraine_rate_last7` rises from rank 3 (0.0156) to
rank 1 (0.0379) on the headache stratified cell, a 2.4-fold magnitude
shift. SHAP therefore corroborates the leakage channel by an
independent method: the stratified-split optimism observed in §1 is a
feature-channel leak through the history features, not a model artefact.
The same Addition recovers the set of Park 2016 trigger factors on the
migraine/park cell with the correct effect direction (all positive ALE
slopes for stress, hormonal change, noise, alcohol, overeating, travel),
though the within-set rank does not align with Park's univariate odds
ratios (Spearman ρ = +0.257 TabPFN headline; ρ = -0.429 XGBoost runner-up;
neither distinguishable from zero at n = 6 shared triggers).

Addition 3 quantifies the temporal dependence directly. The pooled lag-1
autocorrelation of the daily attack series is 0.267 for migraine and
0.218 for headache; a trigger-controlled discrete-time self-excitation
logistic regression gives a lag-1 odds ratio of 5.43 for migraine
(likelihood-ratio test vs triggers-only χ² = 155, p = 2.5e-33) and 2.44
for headache (χ² = 203, p = 1.1e-43); the first-order Markov transition
P(attack tomorrow | attack today) is 0.315 for migraine versus 0.050
without (risk ratio 6.3, χ² = 298) [houle2005timeseries, p. 445;
barra2020markov, p. 3]. The Goh-Barabasi burstiness B is near zero
(median -0.05 migraine, -0.06 headache), so the dependence is
short-range day-to-day clustering rather than long-tailed bursty
memory [goh2008burstiness, p. 2]. Addition 3 therefore licences a
sequence-model test but predicts that any gain will be small because
the recoverable signal sits in a short window the engineered lag/rolling
features already encode.

Addition 4 tests that prediction directly. Across three sequence
variants (window-MLP, GRU, TCN) on the full_features chronological
cells, no sequence variant beats the XGBoost stack headline AUROC; on
migraine the window-MLP narrowly beats the vanilla TabPFN runner-up
(0.771 vs 0.761) but does not unseat the XGB-HP020 0.793 headline. The
verdict is "competitive but not dominant" rather than a uniform null —
the short-range temporal-signal prediction of Addition 3 is supported.
Explicit sequence modelling adds nothing decisive on this cohort once
rolling and lag history features are present, consistent with the
diary-only-without-wearable regime the next-day-diary literature
already occupies [faisal2026forecasting, p. 1].

Addition 5 is the load-bearing measurement. The within-person
C-statistic clusters near 0.55 across the canonical leaves, the
per-patient AUROC distribution centres near chance, and the
pooled-minus-within gap is large — the same gap the Holsteen 2020
argument predicts [holsteen2020triggers, p. 2364]. Three personalisation
regimes (pooled logistic regression, per-patient logistic regression,
partial-pool logistic regression with a per-patient random intercept)
are tested. The partial-pool regime improves the *pooled* AUROC on the
sparse park trigger set (+0.10 over pooled on hold-out), but the gain
does not survive within-person re-evaluation: it is the between-patient
*level* (base-rate) effect, not within-patient day-to-day discrimination.
The leave-one-site-out external check across Uijeongbu and Dongtan
reproduces the within-person near-chance result and reveals
substantial calibration drift (O:E ratio 0.50-1.54 for migraine) driven
by the 8.6% vs 5.7% base-rate gap between the two sites
[huang2020calibration, p. 621].

Addition 6 converts the discrimination layer into a clinical-value
layer. Brier skill against per-patient TRAIN-set climatology is
negative for migraine across all three architectures (XGBoost -0.068,
TabPFN -0.054, sequence -0.117) and positive for headache (+0.139,
+0.210, +0.139); decision-curve net benefit for migraine sits near
zero across the clinically plausible threshold band, while the
headache curve is value-positive at +0.05 to +0.20
[vickers2019dca, p. 1; murphy1993forecast, p. 281]. The
quality-versus-value distinction is therefore target-specific on this
cohort: the migraine forecast fails the Murphy test, the headache
forecast does not.

## 7.3 Implications for the next-day-diary migraine-forecasting field

Three implications follow.

First, pooled AUROC alone is not a sufficient endpoint for a
next-day-diary migraine-forecasting paper. A paper that reports only
pooled discrimination on a small cohort cannot distinguish the
between-patient base-rate-separation component from the within-person
day-to-day component, and the clinically-useful component is the latter.
The reporting standard the prediction-model literature has converged on
— calibration alongside discrimination, decision-curve net benefit
where a deployment decision is implied, and within-person evaluation
for individualised forecasts — should be the floor, not the ceiling,
for this subfield [collins2024tripodAI, p. 6; vickers2019dca, p. 1;
moons2025probastAI; holsteen2020triggers, p. 2364].

Second, the diary-only forecasting regime is intrinsically harder than
the wearable-augmented regime its high-water-mark results occupy. Faisal
2026 reaches AUC 0.84 on 21,550 headache days with trapezius EMG, HRV,
and skin-temperature inputs that the diary-only SHD cohort does not
carry [faisal2026forecasting, p. 1; p. 5]; the diary-only comparator
sits at AUC 0.56-0.73 [houle2017stress, p. 1041; holsteen2020triggers,
p. 2364]. The Park-cohort result reported here lands inside the
diary-only band and confirms it. Reading the Faisal 2026 number as the
diary-only target sets up an unrealistic expectation; the diary-only
benchmark needs its own published reference points, and this work
provides them.

Third, the Park 2016 SHD dataset is the largest open trigger-diary
cohort in the public record but is almost entirely unexploited for
machine learning. Of the 122 papers that cite Park 2016, only one
published ML reuse exists (in a high-school research journal, on a
same-day random-split task that is not next-day forecasting); the
canonical 2025 Cephalalgia review of the ML-migraine-prediction field
[dumkrieger2025review, p. 1] does not cite Park 2016 anywhere. The
benchmark presented here therefore establishes a peer-reviewed
reference point on the open dataset that the field has been overlooking,
in a regime where data scarcity prevents most newer datasets from being
externally validated. The same data-scarcity finding is the reason the
external check here is geographic (leave-one-site-out within Park's two
recruitment clinics) rather than across-cohort: no comparable public
dataset exists for true out-of-cohort transport
[steyerberg2016validation, p. 245].

## 7.4 Limitations

The findings hold for the Park 2016 cohort (62 enrolled patients, 63
unique patient IDs after engineering, ICHD-3 episodic migraine, 19-55
years, 82% female, two Korean university hospitals); the claims do not
extend beyond that applicability domain.

- **Out-of-cohort transport is not claimed.** The Addition 5
  leave-one-site-out check is *within* the Park cohort across two
  enrolment sites; no external dataset is evaluated. No reported
  AUROC, AUPRC, calibration slope, or net-benefit value is asserted to
  generalise to a non-Park cohort, to a different diagnostic
  instrument, or to a wearable-input pipeline.
- **24-hour forecast horizon is operational, not optimal.** The
  next-day target matches the diary's recording resolution and the
  resolution at which oral pre-emptive intervention is usable; other
  horizons (12 h, 48 h, multi-day) are not evaluated.
- **Chronological splits cross patient boundaries.** Train, validation,
  and test sets are partitioned by date, not by patient, so the same
  patient appears in multiple splits. The patient-held-out splits in
  Addition 0 (split type `patient`) provide the held-out-patient read
  alongside the chronological headline; the chronological headline
  AUROC should not be read as a held-out-patient performance estimate.
- **The 7-day rolling-feature window is not ablated.** The 7-day
  rolling-rate and lag-7 features are present by analogy with the
  Park-cohort attack-rate literature; a sweep of window length (e.g.
  3, 5, 14 days) is not performed. The qualitative finding that history
  features dominate the SHAP ranking is robust to the precise window
  because every rolling-window variant is constructed from the same
  source diary days.
- **The TabPFN headline is a family-level claim.** The composite_sorted
  rule's calibration-distance tiebreak picks v2.6 from the AUROC-tied
  trio (v2.6 / v3-default / v3-binary clustered at 0.652-0.653 within
  the rule's 0.02 noise tier); the specific within-family variant
  choice is calibration-driven, not discrimination-driven. The
  reportable cross-architecture claim is at the family level.
- **Within-person personalisation is not claimed to work.** The
  per-patient AUROC distribution and the within-person C-statistic
  cluster near chance, and the partial-pool regime's pooled-AUROC gain
  does not survive within-person re-evaluation. The personalisation
  result is read as a negative finding, not as a basis for clinical
  deployment.
- **ShapIQ interaction magnitudes are sensitive to library versioning
  on the high-dimensional sampling-budget cell.** The exhaustive
  2⁶ = 64-coalition migraine/park cell reproduces within ~1% across
  runs; the 96-coalition-sampling headache/full cell shifts magnitude
  scale ~1.5× across TabPFN/`shapiq`/NumPy version changes. The
  paper's interaction claim is therefore confined to the exhaustive
  cell and reported at the set + direction level, mirroring the
  set + direction discipline of the Park-OR recovery in Claim 2.

## 7.5 Future work

The most direct next step is true out-of-cohort external validation.
The Park-group expanded cohort (Cho 2018 / Park 2018, 82-patient SHED)
is the closest data-lineage match and is access-gated rather than
open; an author-data request via that channel is the only realistic
path to a separate-cohort evaluation. Beyond that, augmenting the
diary modality with wearable signals
[faisal2026forecasting, p. 1; siirtola2018sleep, p. 1] is the field's
established route to higher within-person discrimination but requires
a different data-collection design than the SHD diary supports. Within
the existing modality, a per-patient sequence model (one network per
patient) was de-scoped here because per-patient day counts on the SHD
cohort (median ~70 calendar days, minimum <30) are too small to train
a sequence backbone; a multi-cohort pooled deep-personalisation
approach using transfer from auxiliary diary datasets would be the
methodologically appropriate extension if a comparable open cohort
becomes available.
