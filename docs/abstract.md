# Title and abstract

> Position in the paper: **manuscript front-matter (Title + Abstract +
> Funding)**. Carried into the submission template. The Title satisfies
> TRIPOD+AI Item 1 (identifies the paper as developing/evaluating a
> multivariable prediction model, names the population, the outcomes, and
> the time horizon). The 4-heading structured Abstract follows *Journal of
> Headache and Pain* submission requirements and addresses the 13 TRIPOD+AI
> for Abstracts sub-items [collins2024tripodAI, wt1 p. 1]. The Funding line
> below the Abstract satisfies TRIPOD+AI Item 18a.

## Title

Within-person versus pooled discrimination in next-day migraine and
headache prediction models on the Park 2016 Korean SHD cohort

## Abstract

### Background

Pooled AUROC for daily migraine forecasting conflates within-person day-to-day ranking with between-patient base-rate separation, leaving clinical utility unresolved. Diary-only literature reaches AUROC 0.56-0.73 within-person, below wearable-augmented work (up to 0.84) on physiological inputs the diary lacks.

### Methods

Next-day migraine and headache forecasting were evaluated on the Park 2016 SHD cohort (62 patients; 4,516 patient-days; 7.2% positive; two Korean clinics; 19-55 y; 82.3% female; single ethnicity). Three architectures were compared across three feature sets, three split types (chronological, stratified, patient hold-out), and two train/calibration ratios (70/30 and 70/15/15): XGBoost stacking (with and without 500-trial Optuna HPO), TabPFN (five variants, no per-variant HPO), and a window-MLP sequence baseline. Missing days were structural; no imputation. Headline cells were selected by a composite rule gating on calibration slope; the migraine `full_features` cell carries EPV 5.5 against the Riley/Ensor/Martin criterion (high-risk overfitting band), pre-declared under-powered at the development stage rather than an evaluation-time finding. External validation was leave-one-site-out across the two participating clinics (geographic internal-external). Discrimination, calibration, and bootstrap 95% CIs were reported together; cross-architecture comparisons used paired DeLong with Benjamini-Hochberg correction across 18 documented contrasts (scope frozen at Git tag `analysis-plan-frozen`), escalated to a 166-pair all-pairs Bonferroni sensitivity at the canonical headline cells (0 of 166 significant).

### Results

The within-person C-statistic clustered at 0.53-0.57 across architectures and targets (TabPFN headache 0.542 [0.509-0.575] Paule-Mandel pooled, 57/63 estimable; TabPFN migraine 0.565 [0.508-0.622], 19/63 at the five-positive floor); per-patient AUROC near chance. Chronological-split pooled headline AUROC reached 0.793 (95% CI 0.701-0.873) for migraine (XGB-HP020) and 0.652 (0.589-0.712) for headache (TabPFN-v2.6, headline-tied with v3-default and v3-binary within a 0.02 AUROC tier, picked by calibration-distance tiebreak); pooled AUROC therefore exceeded within-person C by a margin attributable to between-patient base-rate separation, not day-to-day ranking. Headline calibration slopes (patient-cluster primary) were 1.386 [0.401, 2.067] (migraine) and 1.094 [0.594, 1.429] (headache). Brier skill against per-patient TRAIN-set climatology under patient-cluster bootstrap (primary) was significantly positive for headache only on TabPFN-v2.6 (+0.215 [+0.017, +0.396]); the XGBoost and sequence headache Brier skill CIs cross zero under patient-cluster though they were significantly positive under patient-day-iid bootstrap (sensitivity), and migraine Brier skill was significantly negative only on the sequence baseline (-0.117 [-0.241, -0.026] patient-cluster); migraine XGBoost and TabPFN CIs included zero. Leave-one-site-out validation showed calibration drift: observed-to-expected ratio fell to 0.50 when an 8.6%-prevalence site trained the model applied to the 5.7% site, and rose to 1.54 in the opposite direction.

### Conclusion

Pooled AUROC is not a sufficient endpoint for the migraine target on diary-only data; under patient-cluster bootstrap, positive Brier skill survived only for headache at the TabPFN-v2.6 headline cell, and no architecture achieved positive Brier skill for migraine. This work reports development and internal validation on a single 62-patient Korean cohort; we do not recommend clinical deployment.

## Funding

This work received no external funding.

---

**Word count.** Title 18 words. Abstract 449 words (under the JHP 450-word cap; CI-density + demographic-narrowness disclosures per writing_guide §10.1 and §10.11, plus patient-cluster bootstrap as the primary CI unit on AUROC / slope / CITL / Brier skill (patient-day-iid as sensitivity) and the headache-vs-migraine target-asymmetry sentence in the Conclusion).

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
