# Title and abstract

> Position in the paper: **manuscript front-matter (Title + Abstract +
> Funding)**. Carried into the submission template. The Title satisfies
> TRIPOD+AI Item 1 (identifies the paper as developing/evaluating a
> multivariable prediction model, names the population, the outcomes, and
> the time horizon). The 5-heading structured Abstract follows *Journal of
> Headache and Pain* submission requirements and addresses the 13 TRIPOD+AI
> for Abstracts sub-items [collins2024tripodAI, wt1 p. 1]. The Funding line
> below the Abstract satisfies TRIPOD+AI Item 18a.

## Title

Within-person versus pooled discrimination in next-day migraine and
headache prediction models on the Park 2016 SHD cohort

## Abstract

### Objective

To evaluate whether pooled AUROC for 24-hour migraine and headache
forecasting reflects clinically usable within-person day-to-day ranking
or between-patient base-rate separation, on the publicly released Park
2016 smartphone-headache-diary cohort.

### Background

Pooled AUROC for daily migraine forecasting conflates within-person
day-to-day ranking with between-patient base-rate separation, leaving
clinical utility unresolved. The diary-only literature reports AUC in
the 0.56 to 0.73 range under within-person or leave-one-out evaluation;
wearable-augmented forecasts reach up to 0.84 using physiological
signals that diary-only inputs do not carry. Diary-based forecasting
remains a candidate foundation for pre-emptive oral medication.

### Methods

Next-day migraine and headache forecasting were evaluated on the Park
2016 SHD cohort (62 enrolled patients; 4,516 patient-days; 7.2 % pooled
positive rate; two Korean university clinics). The factorial grid spanned
three feature sets, three split types (chronological, stratified,
held-out-patient), two split ratios (70/30 and 70/15/15), and the
architecture families {XGBoost stacking with and without 500-trial Optuna
hyperparameter search; TabPFN tabular foundation model across five
released variants; window-MLP sequence baseline}. A gap-aware reindexer
treated missing diary days as structural, not missing-at-random;
no imputation was applied. Headline cells were selected by
a composite rule gating on calibration slope before breaking AUROC and
AUPRC ties on fixed tier bins. External validation was leave-one-site-out
across the two recruitment clinics. Discrimination, calibration, and
1,000-iteration bootstrap 95 % confidence intervals were reported
together; cross-architecture comparisons used paired DeLong tests with
Benjamini-Hochberg false-discovery-rate correction across 18 pre-registered
contrasts.

### Results

On chronological splits the headline AUROC reached 0.793 for migraine
(XGB-HP020 stack) and 0.652 for headache (TabPFN family v2.6, v3-default,
and v3-binary tied within the composite rule's 0.02 AUROC tier). Median
calibration slope was 0.64. The precision-weighted within-person
C-statistic, estimated by out-of-fold cross-validation on the
non-hyperparameter-tuned 70/30 chronological cells, centred near 0.55;
the per-patient AUROC distribution centred near chance, and the
pooled-minus-within gap was large; the pooled AUROC value therefore exceeded
the within-person C by a margin attributable to base-rate separation. Brier skill against
per-patient climatology was negative for migraine and positive for
headache. The migraine decision-curve net benefit sat near zero across
the clinically plausible threshold band, whereas the headache curve was
value-positive (+0.05 to +0.20). Leave-one-site-out external validation
showed substantial calibration drift (observed-to-expected ratio 0.50 to
1.54) tracking the inter-site base-rate gap.

### Conclusion

A clinically usable 24-hour migraine forecast on this cohort requires
richer per-patient signal or a re-framed prediction target; pooled AUROC
is not a sufficient endpoint for the migraine target on diary-only data.
The finding is migraine-specific: headache forecasts retained
probabilistic value across the same threshold band.

## Funding

This work received no external funding.

---

**Word count.** Title 17 words. Abstract 409 words (Objective 29;
Background 58; Methods 145; Results 131; Conclusion 46). Under the JHP
450-word abstract cap with 41-word headroom.

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
