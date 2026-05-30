# Title and abstract

> Position in the paper: **manuscript front-matter (Title + Abstract +
> Funding)**. Carried into the submission template. The Title satisfies
> TRIPOD+AI Item 1 (identifies the paper as developing/evaluating a
> multivariable prediction model, names the population, the outcomes, and
> the time horizon). The 4-heading structured Abstract follows *Journal of
> Headache and Pain* submission requirements and addresses the 13 TRIPOD+AI
> for Abstracts sub-items [7]. The Funding line
> below the Abstract satisfies TRIPOD+AI Item 18a.

## Title

Within-person versus pooled discrimination in next-day migraine and
headache prediction models on the Park 2016 Korean SHD cohort

## Abstract

### Background

Pooled AUROC for daily migraine forecasting conflates within-person day-to-day ranking with between-patient base-rate separation, leaving clinical utility unresolved. Diary-only literature reaches AUROC 0.56-0.73 within-person, below wearable-augmented work (up to 0.84) on inputs the diary lacks.

### Methods

Next-day migraine and headache forecasting were evaluated on the Park 2016 SHD cohort (62 patients; 4,516 patient-days; 7.2% positive; two Korean clinics; 19-55 y; 82.3% female; single ethnicity). Three architectures (XGBoost stacking ±500-trial Optuna HPO; TabPFN, five variants, no per-variant HPO; window-MLP sequence baseline) × three feature sets × three split types (chronological/stratified/patient hold-out) × two ratios (70/30, 70/15/15) were evaluated. Missing days were structural; no imputation. Headline cells used a composite rule gating on calibration slope; the migraine `full_features` cell carries EPV 5.5 (high-risk, pre-declared under-powered). Leave-one-site-out across both clinics (geographic internal-external). Discrimination, calibration, and bootstrap 95% CIs; cross-architecture paired DeLong + BH-FDR across 18 documented contrasts, with a 166-pair all-pairs Bonferroni sensitivity at headline cells (0 of 166 significant).

### Results

The within-person C-statistic clustered at 0.53-0.57 across architectures and targets (headache 0.542 [0.509, 0.575] Paule-Mandel pooled, 57/63 estimable; migraine 0.565 [0.508, 0.622], 19/63 at the five-positive floor); per-patient AUROC near chance. Pooled headline AUROC reached 0.793 (95% CI 0.701-0.873) for migraine (XGB-HP020) and 0.652 (0.589-0.712) for headache (TabPFN-v2.6); pooled AUROC therefore exceeded within-person C from between-patient base-rate separation, not day-to-day ranking. Headline calibration slopes (patient-cluster primary) were 1.386 [0.401, 2.067] (migraine) and 1.094 [0.594, 1.429] (headache). Brier skill against per-patient TRAIN-set climatology (patient-cluster primary) was significantly positive for headache only on TabPFN-v2.6 (+0.215 [+0.017, +0.396]); XGBoost and sequence headache CIs crossed zero under patient-cluster, and migraine Brier skill was significantly negative only on the sequence baseline (-0.117 [-0.241, -0.026]); migraine XGBoost and TabPFN CIs included zero. Leave-one-site-out calibration drift: observed-to-expected ratio ranged 0.50-1.54 across the two clinics.

### Conclusion

Pooled AUROC is not a sufficient endpoint for migraine on diary-only data; positive Brier skill survived only for headache at the TabPFN-v2.6 headline cell, and no architecture achieved positive Brier skill for migraine. Development and internal validation on a single 62-patient Korean cohort; no clinical deployment recommended.

## Keywords

Migraine; Headache; Forecasting; Machine learning; Calibration; Decision curve analysis; Smartphone applications; Within-person evaluation

## Funding

This work received no external funding.

---

**Word count.** Title 18 words. Abstract 340 words (under the JHP 350-word research-article cap; CI-density + demographic-narrowness disclosures per writing_guide §10.1 and §10.11, with patient-cluster bootstrap as the primary CI unit on AUROC / slope / Brier skill and the headache-vs-migraine target-asymmetry sentence retained in the Conclusion).

**TRIPOD+AI for Abstracts sub-items covered.** Title (Item 1), Objective
(Background paragraph), Setting and participants (Methods; cohort, sites,
sample sizes), Predictors (Methods; feature sets), Outcomes (Title +
Methods; next-day migraine and headache), Sample size (Methods;
4,516 patient-days), Statistical methods (Methods; factorial grid,
composite selection rule, bootstrap CIs, paired DeLong + BH-FDR),
Missing-data handling (Methods; structural-not-MAR, no imputation),
Performance measures (Results; discrimination, calibration, decision
curve, Brier skill), External validation (Methods + Results;
leave-one-site-out, O:E drift), Results (Results paragraph; AUROC,
within-person C, decision-curve net benefit), Limitations / implications
(Conclusion; target-specific finding), Funding (Funding paragraph).
