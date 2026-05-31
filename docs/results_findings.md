# Results findings (Additions 0-1, full grid incl. patient hold-out)

> Position in the paper: **Results (5)**. Reads after `addition6_clinical_value.md`; precedes `addition2_results.md`. Cross-Addition synthesis; the canonical entry point for the paper's findings, ahead of the per-Addition results docs.


Data-verified findings from the aggregated sweep of 488 leaf cells
(comparison CSV of run 20260526T003803). Each claim below is backed by
the per-cell hold-out metrics; the verifying computation is noted so it
is reproducible. These are the empirical results the paper draws on; the
methods rationale lives in the per-addition docs.

**On the "Addition" naming.** The Additions are not a strict sequence
where N+1 is gated on N. They are a parallel decomposition of the
benchmark question into independent slices: Additions 0 and 1 are the
two tabular architecture families (XGBoost stack, TabPFN foundation
model) compared head-to-head on the same feature sets and splits;
Addition 2 layers explainability on top of those headline leaves;
Addition 3 tests whether the IID-row assumption Additions 0 and 1
implicitly make is justified by the temporal structure of the cohort;
Addition 4 fits sequence models on the same feature stream as a
direct cross-check of Addition 3's IID verdict; Addition 5
characterises within-patient generalisation and external-site
transport of the resulting models; Addition 6 reports the clinical
value (decision curve, Brier skill) of the same models against
trivial-baseline forecasts. Each Addition is a piece of evidence the
paper draws on independently, so the per-Addition conclusions stand
or fall on their own data, not on a sequential chain.

## 1. Stratified-split optimism was driven by feature-channel leakage, not by the model

A tabular model (XGBoost stack, TabPFN) was row-permutation-invariant and
it cannot learn sequence or row order. So a random/stratified split
cannot leak "temporal structure" through the model. The optimism a
stratified split shows on this cohort is therefore mediated entirely by
the **lag/rolling feature values**, which carry information about a
patient's neighbouring days; when random shuffling places those
neighbouring days on both sides of the train/test boundary, the test
row's lag features encode train-set outcomes.

This predicted the leak was **feature-set-dependent**, and the data
confirm it. Holding the architecture fixed (stacked_2xgb, NonHP),
mean hold-out AUROC over ratios, stratified minus chronological:

| feature set | lag/rolling features | headache strat-chrono | migraine strat-chrono |
|---|---|---|---|
| full       | 17 | +0.110 | +0.038 |
| spano      | 14 | +0.093 | +0.030 |
| no_rolling | 0  | -0.034 | -0.035 |
| park       | 0  | -      | -0.028 |

Feature sets with lag features (full, spano) show inflated stratified
AUROC; feature sets without them (no_rolling, park) show no inflation -
stratified is in fact slightly *worse* than chronological, consistent
with pure sampling noise and the absence of a leakage channel. The leak
is localised to the feature channel, exactly as the
permutation-invariance argument predicts.

Practical consequence: random/stratified splitting is only an
optimistic-bias risk for feature sets that contain history features.
The chronological split remains the honest evaluation, and for the
no-history feature sets the stratified-vs-chronological difference
reflects interpolation (same time period) versus temporal extrapolation
(future), not leakage.

Verify: `holdout_AUROC` grouped by (target, feature_set, splittype)
within `architecture==stacked_2xgb_meta_lr` and `hp_strategy` null.

## 2. The split-difficulty hierarchy is target-dependent

There is no single "stratified > patient > chrono" ordering; it depends
on the target's base rate (stacked_2xgb NonHP, full_features, mean over
ratios):

- **Headache** (~23% positive): stratified 0.670 > patient 0.632 >
  chrono 0.560. Patient generalisation sits between leakage-inflated
  stratified and temporally-hard chronological.
- **Migraine** (~7% positive): stratified 0.698 > chrono 0.660 >
  **patient 0.550**. Whole-patient hold-out is the *hardest* split for
  the sparse target; generalising to an unseen patient with few
  positive days is harder than temporal extrapolation.

The paper should report the hierarchy per target rather than as a
single ordering.

## 3. Architecture comparison (chronological cells only, full_features)

On the trustworthy chronological cells:

- **Headache**: TabPFN and the XGBoost stack are near-identical
  (stacked_2xgb 0.560; TabPFN versions 0.570-0.575, edge +0.01-0.015).
  The large TabPFN advantage seen in the across-all-splits CD diagram is
  partly driven by the leaky stratified cells; on chronological cells
  alone the architectures are close.
- **Migraine**: AutoTabPFN (post-hoc ensemble, the four promising leaves
  only) leads clearly at 0.745 vs stacked_2xgb 0.660 (+0.085); the
  vanilla single-fit TabPFN versions sit slightly *below* the XGBoost
  stack (0.619-0.635). So the migraine story is AutoTabPFN > stacked_2xgb
  > vanilla TabPFN, not "TabPFN beats XGBoost" wholesale.

## 3a. Paired DeLong + BH-FDR: cross-architecture significance

The qualitative "close" calls in §3 are tested formally with paired
DeLong [delong1988comparing] on 18 cell-pair tests (same target,
feature set, split, and ratio so the same test instances are used for
both architectures), with Benjamini-Hochberg FDR correction
[benjamini1995controlling] across the full test family at q ≤ 0.05.
The test family covers three scientific questions: (a) cross-family
headline vs runner-up at each (cell, split) where the composite_sorted
selection produces a same-ratio pair; (b) within-family-tie tests at
the headache full chrono 70_30 cell where v2.6 / v3-default / v3-binary
sit within the 0.02 AUROC composite tier; (c) the HP-ladder at the
migraine full chrono 70_30 cell (HP020 vs HP050 / HP100 / HP200 /
HP500) and the AutoTabPFN-vs-XGBoost comparison at the same cell -
both required to test §3's "AutoTabPFN leads migraine" and
"HP020 is the composite winner" claims.

| cell | ratio | arch A | arch B | AUROC A | AUROC B | ΔAUC | p (DeLong) | q (BH) | sig |
|---|---|---|---|---|---|---|---|---|---|
| headache/full/chrono | 70_30 | tabpfn_v2-6 | stacked_2xgb_NonHP | 0.653 | 0.640 | +0.013 | 0.5734 | 0.9095 | no |
| headache/no_rolling/chrono | 70_30 | tabpfn_v3-binary | stacked_2xgb_NonHP | 0.582 | 0.554 | +0.028 | 0.2848 | 0.8468 | no |
| migraine/full/chrono | 70_30 | stacked_2xgb_HP020 | tabpfn_v2-5-finetuned | 0.791 | 0.761 | +0.030 | 0.0576 | 0.3454 | no |
| migraine/park/chrono | 70_30 | tabpfn_v3-binary | stacked_2xgb_NonHP | 0.649 | 0.627 | +0.021 | 0.4093 | 0.8468 | no |
| migraine/park/patient | 70_30 | stacked_2xgb_NonHP | tabpfn_v3-default | 0.540 | 0.536 | +0.004 | 0.6906 | 0.9095 | no |
| migraine/no_rolling/patient | 70_30 | tabpfn_v3-binary | stacked_2xgb_NonHP | 0.600 | 0.496 | +0.103 | 0.0020 | 0.0365 | **yes** |
| headache/full/chrono | 70_30 | tabpfn_v2-6 | tabpfn_v3-default | 0.653 | 0.654 | -0.001 | 0.9260 | 0.9535 | no |
| headache/full/chrono | 70_30 | tabpfn_v2-6 | tabpfn_v3-binary | 0.653 | 0.653 | -0.000 | 0.9535 | 0.9535 | no |
| headache/full/chrono | 70_30 | tabpfn_v3-default | tabpfn_v3-binary | 0.654 | 0.653 | +0.000 | 0.8943 | 0.9535 | no |
| headache/no_rolling/stratified | 70_15_15 | tabpfn_v2-5-real | stacked_2xgb_NonHP | 0.570 | 0.576 | -0.005 | 0.7241 | 0.9095 | no |
| migraine/full/patient | 70_15_15 | tabpfn_v3-binary | stacked_2xgb_HP020 | 0.716 | 0.686 | +0.031 | 0.1809 | 0.6513 | no |
| migraine/full/stratified | 70_15_15 | tabpfn_v2-5-finetuned | autotabpfn_v2-5-auto | 0.735 | 0.732 | +0.003 | 0.6662 | 0.9095 | no |
| migraine/full/chrono | 70_30 | stacked_2xgb_HP020 | stacked_2xgb_HP050 | 0.791 | 0.793 | -0.002 | 0.7579 | 0.9095 | no |
| migraine/full/chrono | 70_30 | stacked_2xgb_HP020 | stacked_2xgb_HP100 | 0.791 | 0.787 | +0.004 | 0.4705 | 0.8468 | no |
| migraine/full/chrono | 70_30 | stacked_2xgb_HP020 | stacked_2xgb_HP200 | 0.791 | 0.787 | +0.004 | 0.4705 | 0.8468 | no |
| migraine/full/chrono | 70_30 | stacked_2xgb_HP020 | stacked_2xgb_HP500 | 0.791 | 0.787 | +0.004 | 0.4705 | 0.8468 | no |
| migraine/full/chrono | 70_30 | autotabpfn_v2-5-auto | stacked_2xgb_HP020 | 0.744 | 0.791 | -0.047 | 0.0406 | 0.3454 | no |
| migraine/full/chrono | 70_30 | autotabpfn_v2-5-auto | stacked_2xgb_NonHP | 0.744 | 0.775 | -0.031 | 0.1753 | 0.6513 | no |

(AUROC values are point estimates on the locked test set, which is the
input the paired DeLong test statistic uses; they differ from the
bootstrap-mean AUROC values reported in §3 by 0.001-0.002 due to
bootstrap distribution asymmetry on small positive counts.)

Five findings.

**(1) The "architecture comparison is the smallest source of variation"
framing is statistically supported.** Of nine cross-family comparisons
across cells and ratios, eight are not distinguishable from zero at
q ≤ 0.05 after BH-FDR correction. The migraine chronological headline
(HP020 vs TabPFN-finetuned, ΔAUC +0.030) is borderline raw (p = 0.058)
but does not survive FDR (q = 0.35). The qualitative "close" calls in
§3 are backed by formal tests.

**(2) The headache full chrono family-level tie is statistically
defensible.** All three within-family TabPFN comparisons
(v2.6 / v3-default / v3-binary, AUROC clustered at 0.653-0.654) are
non-significant with q ≥ 0.95, ΔAUC magnitudes ≤ 0.001. The
composite_sorted rule's calibration-distance tiebreak is the
appropriate selection mechanism here; the within-family rank ordering
carries no statistical signal.

**(3) The HP-search budget is statistically flat on this cell.** HP020
vs HP050 / HP100 / HP200 / HP500 all give ΔAUC in [-0.002, +0.004], all
non-significant at q ≥ 0.85. The composite-rule's pick of HP020 as the
migraine headline is choosing the lowest search budget within a tier
of statistically equivalent results; higher-budget search neither
helps nor hurts (corroborating §5's "the single-objective search
saturates by ~100 trials" claim by independent measurement).

**(4) AutoTabPFN does not beat HP-tuned XGBoost at the canonical
migraine headline cell.** On the load-bearing migraine/full/chrono/70_30
cell, AutoTabPFN (AUROC 0.744) vs XGB-HP020 (AUROC 0.791) gives
ΔAUC = -0.047 in favour of HP020 (p = 0.041 raw, q = 0.35 after FDR);
AutoTabPFN vs XGB-NonHP gives ΔAUC = -0.031 (p = 0.175, q = 0.65). The
§3 prose comparison "AutoTabPFN leads clearly at 0.745 vs stacked_2xgb
0.660" reads single-cell AutoTabPFN (0.745 at chrono 70_30) against an
across-ratio-mean of NonHP-XGB (the chrono cells average across
70_15_15, 70_30, 80_20 to 0.660, dragged down by the 70_15_15 and
80_20 cells where NonHP sits at 0.610 and 0.592 respectively). At the
same-cell paired test, NonHP-XGB sits at 0.777 (well above AutoTabPFN's
0.745). The within-cell paired tests therefore show no statistical
AutoTabPFN advantage and indicate the §3 framing is asymmetric on the
NonHP side.

**(5) The one paired difference that survives FDR is on the small/sparse
patient hold-out cell.** On migraine/no_rolling/patient/70_30, TabPFN
v3-binary (AUROC 0.600) beats stacked_2xgb NonHP (AUROC 0.496,
anti-predictive) by ΔAUC = +0.103, p = 0.002, q = 0.036. This is the
only quantitatively significant cross-architecture difference in the
18-test family, and it lies exactly where §4 predicted: TabPFN's
single-forward-pass in-context inference is robust where the XGBoost
stack collapses on small leakage-prone splits.

The full 18-row JSON output is persisted at
`experiment/_eval/paired_delong_<timestamp>.json`; the family is
rendered as a forest plot (Figure G6) and as a BH-FDR significance
heatmap (Figure G8) for visual inspection.

## 3b. Exhaustive supplementary: Bonferroni FWER across the all-pairs cross-architecture grid

The §3a 18-test BH-FDR family targets specific hypotheses. As a
*supplementary* counterpart, an all-pairs within-cell paired-DeLong
analysis at ratio 70_30 was run with Bonferroni FWER correction -
strictly more conservative than BH-FDR, and the standard "fishing
defense" inferential procedure. Scope: 15 (target, feature_set, split)
cells where ≥2 architectures share the same test set; 487 paired tests
total; Bonferroni α* = 1.03 × 10⁻⁴.

The pattern of significance is *internally coherent with the §4
robustness narrative*:

| Cell pattern | Bonferroni-sig pairs | Total pairs | What it says |
|---|---:|---:|---|
| **Chronological** (canonical paper-headline cells) | 0 | 166 | Architectures are **exhaustively** indistinguishable at the honest cells |
| **Patient hold-out** | 54 | 125 (43%) | TabPFN robust where XGB-HP collapses to anti-predictive |
| **Stratified** (leakage-inflated) | 41 | 196 (21%) | Architecture-dependent leakage response |

The 0-of-166 chronological result is the load-bearing one for §3's
family-level-tie framing: under the strictest correction in the
all-pairs family, no cross-architecture pair at the canonical headline
cells is statistically distinguishable. The 95-of-321 (29%) pattern on
the hard cells (patient + stratified) corroborates the §4 claim that
TabPFN's robustness on small/leakage-prone splits is reproducible
under exhaustive scrutiny.

The headline §3a finding (migraine/no_rolling/patient TabPFN-v3-binary
vs XGB-NonHP, BH-FDR q = 0.036) appears in the exhaustive family as
ΔAUC = +0.103 with raw p = 0.002; under Bonferroni α* = 1.03e-4 with
m = 487 it does not survive, the expected power loss when m grows by
27×. The effect is real at the BH-FDR scope where the test family is
hypothesis-targeted; the supplementary Bonferroni is intentionally
strict to bound the fishing-defense interpretation.

The full per-cell results are persisted at
`experiment/_eval/exhaustive_delong_per_cell/<cell>.json` for
individual parseability. The supplementary heatmap is
`docs/methodAndResults_diagramCreatorScripts/figures/fig_g7_significance_heatmap.pdf`.

## 4. Robustness: TabPFN held up on cells where HP-tuned XGBoost stacking collapsed

Of 42 anti-predictive cells (hold-out AUROC < 0.45) across the 488-cell
grid, 38 are `stacked_2xgb_meta_lr`; the remaining four are TabPFN and
all are marginal (0.417, 0.446, 0.448, 0.449). The XGBoost anti-predictive cells
concentrate in (a) HP variants on **stratified** splits (the
documented Platt-inversion where hyperparameter search overfits a leaky
split and the calibrator fits a negative slope), and (b) sparse
**migraine patient hold-out**. TabPFN's single-forward-pass in-context
inference does not exhibit this collapse. This is a robustness argument
for foundation-model tabular inference on small, leakage-prone clinical
splits.

Verify: cells with `holdout_AUROC < 0.45` grouped by architecture and
splittype.

## 5. Hyperparameter tuning: apparent large gains were inflated by selection

Taking the best single_AUROC HP variant vs NonHP on chronological
full_features suggests +0.097 (headache) and +0.135 (migraine) AUROC.
This is a max-over-~15-variants quantity (HP020/050/100/200/500 x ratios
x strategies) compared against a NonHP mean, so it carries a winner's
-curse / multiple-comparisons bias. The honest HP-effect estimate must
compare like with like (best HP vs best NonHP at a fixed ratio, or the
HP distribution with a selection correction), and should be reported
with that caveat. The earlier budget-ladder analysis already showed the
single-objective search saturates by ~100 trials, so a true HP gain of
the magnitude above is not credible without the selection correction.

## 6. Calibration was fragile on the small/sparse cells

Across 488 cells the calibration slope has median 0.64 (over-confident,
the classic small-sample overfitting fingerprint), and 98 cells have a
negative slope (Platt inversion on a tiny calibration sub-split). At
the chronological headline cells the bootstrap CIs are wide: the
migraine *XGB-HP020 / full / chrono / 70-30* leaf has calibration
slope 1.386 [0.401, 2.067] patient-cluster primary and the headache
*TabPFN v2.6 / full / chrono / 70-30* leaf 1.094 [0.594, 1.429]
patient-cluster primary; both CIs include 1.0, so calibration is
not distinguishable from unity at the headline cells, though the
wide CIs reflect the small-sample calibration uncertainty.
The paper must report calibration slope with its bootstrap CI alongside
discrimination and flag that threshold-derived metrics on the
negative-slope cells are unreliable; AUROC/AUPRC remain interpretable
(rank-based).

## 7. Integrity

1 of 488 cells has a missing (NaN) hold-out AUROC; the rest of every
patient leaf with a fitted model has a results file. The one transient scipy
ImportError and the addition-1 2-way `pd`-import bug encountered during
the run were both resolved (retry recovery; template fix + re-evaluate),
so the grid is complete and reproducible from the committed code.

## Headline takeaways for the paper

1. **The pooled AUROC overstates clinical utility, and held-out-patient
   transport collapses on the migraine headline.** On the chronological
   headline cells the migraine AUROC reaches 0.793 [0.701, 0.873]
   (*XGB-HP020 / full / chrono / 70-30*) and the headache AUROC 0.652
   [0.589, 0.712] (*TabPFN family / full / chrono / 70-30*: v2.6 /
   v3-default / v3-binary tied within the composite rule's 0.02 AUROC
   tier). At the **held-out-patient** counterparts (same architecture and
   ratio, split type `patient`), the migraine cell collapses to AUROC
   **0.283** [0.233, 0.333] (anti-predictive, with calibration slope
   -0.748, Platt-inverted), while the headache TabPFN family stays
   within band at 0.631-0.637 (v2.6 0.631 [0.593, 0.664]; v3-default
   0.635 [0.599, 0.669]; v3-binary 0.637 [0.601, 0.672]). The
   chronological 0.793 is a within-cohort deployable number, not a
   held-out-patient generalisation number. Pooled discrimination is
   overwhelmingly between-patient base-rate separation, not within-person
   day-to-day ranking (see takeaway 2).
2. **Within-person personalisation is not demonstrated.** The
   precision-weighted within-person C-statistic clusters near 0.55
   on the canonical leaves (Addition 5); Brier skill against
   per-patient climatology under patient-cluster bootstrap (primary)
   is significantly positive for headache only on TabPFN-v2.6
   (+0.215 [+0.017, +0.396]); the headache XGBoost (+0.198
   [-0.017, +0.385]) and sequence (+0.139 [-0.123, +0.362]) CIs
   cross zero under patient-cluster though both are positive under
   patient-day-iid sensitivity, and only significantly negative for
   migraine on the sequence baseline (-0.117 [-0.241, -0.026]
   patient-cluster). Migraine XGBoost and TabPFN CIs include zero,
   so the tabular forecasts are indistinguishable from the per-patient
   climatology baseline rather than significantly worse (Addition 6).
   The near-chance within-person result replicates across the two
   recruitment sites in the leave-one-site-out validation
   (`docs/external_validation_site.md` Section 7), so it is not a
   single-cohort artefact. High pooled AUROC does not translate into
   within-patient probabilistic value at this sample size; the
   headline numbers should be read as cohort-level discrimination,
   not personalised forecasting skill.
3. Stratified-split inflation is a **feature-channel leak via history
   features**, not a model artefact; proven by the zero inflation on
   no-history feature sets (Section 1). The leakage mechanism is
   corroborated by an independent method: SHAP attribution
   (Addition 2 Claim 1) shows the history features
   (`migraine_rate_last7`, `headache_free_streak`,
   `days_since_last_migraine`) carry disproportionate mean-|SHAP|
   on stratified full_features leaves and a much smaller share on
   the corresponding chronological leaves.
4. Split difficulty is **target-dependent**; patient hold-out is hardest
   for sparse migraine (Section 2).
5. On honest (chronological) cells, **architecture ordering at the
   migraine headline is not statistically resolvable** (Section 3a).
   The composite-sorted winner (calibration-gated) is the
   *XGB-HP020 / full / chrono / 70-30* stack at AUROC 0.793; the
   §3-aggregate "AutoTabPFN leader" comparison (AutoTabPFN 0.745 vs
   stacked_2xgb 0.660 across multiple leaves) does not reproduce as a
   paired within-cell test at the headline cell, where AutoTabPFN
   (0.744) trails HP020 (0.791) by ΔAUC = -0.047 (p = 0.041 raw,
   q = 0.35 after FDR). The HP-search budget is also statistically
   flat at this cell: HP020 vs HP050 / HP100 / HP200 / HP500 all give
   ΔAUC in [-0.002, +0.004] and q ≥ 0.85, so the composite-rule
   pick of HP020 is choosing the lowest search budget within a tier
   of statistically equivalent results. Headache is a near-tie across
   architectures (Section 3) and the headline is best read as a
   family-level claim: paired DeLong on v2.6 / v3-default / v3-binary
   at this cell gives q ≥ 0.95 for every pair (Section 3a), so the
   within-family rank is not statistically resolvable; the variant
   choice is calibration-driven, not discrimination-driven.
6. **TabPFN is more robust** than HP-tuned XGBoost stacking on
   small/leaky splits (Section 4). The only paired difference that
   survives BH-FDR across nine cross-architecture comparisons is on
   migraine/no_rolling/patient/70_30: TabPFN v3-binary AUROC 0.600 beats
   stacked_2xgb NonHP AUROC 0.496 by ΔAUC = +0.103, p = 0.002,
   q = 0.036 (Section 3a), exactly the cell type where Section 4's
   anti-predictive-XGBoost-cells pattern predicts the robustness gap.
7. **Explicit sequence modelling does not dominate the tabular
   baselines** on the honest chronological cells (Addition 4). All
   three sequence variants (window-MLP, GRU, 1D-CNN) trail the XGBoost
   stack headline AUROC; on migraine the window-MLP narrowly beats
   the vanilla TabPFN baseline (0.77 vs 0.76) but does not unseat
   the XGB-HP020 0.793 headline. This is the falsification result the
   Addition-3 short-range-dependence finding predicted: once
   rolling/lag history features are in the feature set, an explicit
   recurrent backbone adds nothing decisive.
8. **Calibration transports badly across sites within the study.**
   Leave-one-site-out validation (`docs/external_validation_site.md`)
   shows discrimination is preserved across the two clinics, but
   calibration-in-the-large drifts substantially (O:E ratio 0.50-1.54
   for migraine), driven by the base-rate gap between sites
   (Uijeongbu 8.6% vs Dongtan 5.7%). A migraine model trained on the
   8.6% site over-predicts when applied to the 5.7% site, and
   vice-versa. This is within-study geographic transport, not
   separate-cohort transport, and PROBAST applicability is bounded
   accordingly.
9. Calibration must be reported and is fragile at this sample size
   (Section 6).

## 8. Limitations and out-of-scope claims

The findings above hold for the Park 2016 SHD cohort (62 enrolled
patients, 63 unique patient IDs after engineering, ICHD-3 episodic
migraine, 19-55 years, 82.3% female, two Korean university hospitals).
The claims we make do not extend beyond that applicability domain.
Specifically:

- **Out-of-cohort transport is not claimed.** No external dataset is
  evaluated (the leave-one-site-out external check in Addition 5 is
  *within* the same Park cohort across the two enrolment sites). We do
  not claim that any reported AUROC, AUPRC, calibration slope, or
  net-benefit value generalises to a non-Park cohort, to a different
  diagnostic instrument, or to a wearable-input pipeline.

- **24-hour forecast horizon is operational, not optimal.** The
  next-day target is the resolution at which the SHD diary records and
  the resolution at which an oral pre-emptive intervention would be
  usable. We do not claim 24 h is the *best* horizon. Other horizons
  (12 h, 48 h, multi-day) are not evaluated here; the prospective
  utility curve over the horizon axis is left for future work.

- **Within-person personalisation is not claimed to work.** Addition 5
  reports per-patient AUROC and the precision-weighted within-person
  C-statistic, and the central finding is that the pooled AUROC is
  largely between-patient base-rate separation: per-patient
  discrimination is near chance. Read this as a negative result on
  personalisation, not a basis for clinical deployment.

- **Chronological splits cross patient boundaries.** Train, val, and
  test sets are partitioned by date, not by patient. Patients appear in
  multiple splits, which means same-patient memorisation through
  history-derived features is a possible source of optimistic
  discrimination. The patient-held-out splits in Addition 0 (split
  type `patient`) provide the held-out-patient discrimination read
  alongside the chronological one; the chronological headline AUROC
  should not be read as a held-out-patient performance estimate.

- **The 7-day rolling-feature window is not ablated.** The 7-day
  rolling-rate and lag-7 features are present by analogy with the
  Park-cohort attack-rate literature; a sweep of the window length
  (e.g. 3, 5, 14 days) is not performed. The SHAP rankings may shift
  with a different window choice; the *qualitative* finding that
  history features dominate the ranking is robust to the precise
  window because every rolling-window variant is constructed from the
  same source diary days.

- **Cross-architecture significance is reported via overlapping
  bootstrap CIs, not via a paired test.** We report the 1000-iteration
  bootstrap 95% CI per metric; pairwise comparisons against an
  architecture-paired test (DeLong for AUROC, or a paired bootstrap
  with FDR control across the 21 cell-by-split groups) are not
  performed. Where the bootstrap CIs overlap we treat the comparison as
  inconclusive rather than significant; this is a conservative but
  weak inferential procedure.

- **Re-identification risk of the 62-patient public cohort is not
  re-litigated.** Park 2016 released the SHD dataset under their
  ethics approval; the *k* < 5 risk from combining hospital site,
  demographic baseline, and dated diary entries is acknowledged but we
  do not re-evaluate that risk. We use the engineered features derived
  from the public release and do not redistribute the raw patient-day
  rows.

- **Headline-leaf selection is composite, and within-family fragile.**
  The five-level `composite_sorted` rule in `experiment/2/select.py`
  uses a 0.02-AUROC tier bucket and a calibration-slope window of
  (0.0, 5.0). A perturbation sweep
  (`experiment/2/sensitivity_composite.py`) shows the cross-architecture
  conclusion is stable: TabPFN wins the headache full_features/chrono
  cell and the XGBoost stack wins the migraine cell under every
  perturbation in {AUROC_TOL = 0.01 / 0.02 / 0.03; CALIB_MIN = -0.5 /
  0.0; CALIB_MAX = 3.0 / 5.0 / 7.0}. The within-family variant choice
  is not stable: the headache TabPFN headline flips between
  version_2-6 (at the default 0.02 tier) and version_2-5-finetuned
  (at the tighter 0.01 tier), and the migraine XGBoost-stack headline
  flips between HP020 (default) and HP050 (at the 0.01 or 0.03 tiers).
  A reader treating the specific variant as load-bearing is therefore
  reading more into the cell label than the data supports; the
  architecture-family conclusion is the more defensible claim. The
  composite ordering is deterministic given the bootstrap seed and
  reproducible from the frozen `figdata_<ts>.json`.
