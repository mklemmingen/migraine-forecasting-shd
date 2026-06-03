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

Within-person vs pooled discrimination, next-day migraine-headache prediction: Park 2016 Korean SHD

## Plain language summary

People with migraine often use smartphone diaries hoping the app can predict tomorrow's attack. We tested whether published accuracy numbers from such diaries actually tell you which of your days will be a migraine day, using a public Korean diary dataset of 62 people. We found that the popular accuracy score (pooled AUROC) mostly reflects differences *between* patients rather than *within* one patient day-to-day, so a diary-only prediction is not yet accurate enough to guide an individual medication decision. Better forecasts will likely need wearable sensors and larger cohorts.

## Abstract

### Objective

To develop and internally validate a multivariable next-day forecasting benchmark for episodic migraine and headache on a publicly available Korean smartphone diary cohort, contrasting pooled with within-person discrimination as the load-bearing endpoint.

### Background

Pooled AUROC for daily migraine forecasting conflates within-person day-to-day ranking with between-patient base-rate separation, leaving clinical utility unresolved. The diary-only literature reaches AUROC 0.56-0.65 within-person, below the wearable-augmented high-water mark (up to 0.84) on inputs the diary lacks.

### Methods

Next-day migraine and headache forecasting were developed and internally evaluated on the Park 2016 SHD cohort (62 patients; 4,516 patient-days; 7.2% positive; two Korean clinics; 19-55 y; 82.3% female; single ethnicity). Three architectures (XGBoost stack, TabPFN six variants, three sequence baselines) × three feature sets × three split types × two ratios were evaluated. Predictors comprised eighteen Park 2016 same-day trigger flags plus engineered history features. Missing days were treated as structural, with no imputation. Headline cells were selected by a composite rule gating on calibration slope; the migraine `full_features` cell carries EPV 5.5 and is pre-declared under-powered. A leave-one-site-out check provided geographic internal-external validation. Bootstrap 95% CIs used patient-cluster resampling. Cross-architecture comparisons used paired DeLong tests with Benjamini-Hochberg FDR correction across 18 documented contrasts. Not registered.

### Results

The within-person C-statistic clustered at 0.53-0.57 across architectures and both targets (headache 0.542 [0.509, 0.575]; migraine 0.565 [0.508, 0.622]); per-patient AUROC centred near chance. Pooled AUROC exceeded within-person C by 0.23 (migraine) and 0.11 (headache), reaching 0.793 (95% CI 0.544-0.890) and 0.652 (0.556-0.740) respectively on the chronological split, which is patient-overlapping by design and tests within-patient temporal generalisation rather than held-out-patient transport; the pooled migraine CI lower bound (0.544) sits barely above chance, so pooled discrimination reflected between-patient base-rate separation rather than within-person day-to-day ranking. Under conservative Bonferroni FWER correction across 166 architecture pairs at the canonical chronological headline cells, 0 contrasts achieved significance; under less-conservative BH-FDR correction across 18 documented contrasts, 1 survived (q ≤ 0.05) at a non-canonical small-cohort patient-hold-out cell. Brier skill against per-patient TRAIN-set climatology was positive for headache only on the TabPFN-v2.6 headline cell (+0.215 [+0.017, +0.396]); migraine Brier skill CIs included or fell below zero across architectures. Headache decision-curve net benefit was positive at clinically plausible low thresholds; migraine net benefit sat near zero. Leave-one-site-out calibration drift was observed across the two clinics.

### Conclusion

Pooled AUROC is not a sufficient endpoint for migraine on diary-only data; positive Brier skill survived only for headache at the TabPFN-v2.6 headline cell, and no architecture achieved positive Brier skill for migraine. Development and internal validation on a single 62-patient Korean cohort; we do not recommend clinical deployment.

## Keywords

Migraine; Headache; Forecasting; Machine learning; Calibration; Mobile applications; Mobile health; Models, Statistical; Within-person evaluation; Internal-external validation

## Funding

This work received no external funding.

---

**Graphical abstract.** Submitted alongside as a supplementary file (`docs/methodAndResults_diagramCreatorScripts/figures/graphical_abstract.png`, 920 x 300 px, 59.5 KB, PNG; rendered by `graphical_abstract.py`). Visualises the pooled-vs-within-person discrimination gap (the four headline anchor numbers from §3.2 + §3.6, against a chance reference at 0.5) with the load-bearing takeaway sentence along the bottom. Per JHP submission guidelines.

**Word count.** Title 11 words, 99 characters (under the JHP ≤ 100-character title soft-cap). Abstract 350 words (under the JHP 350-word research-article cap; CI-density + demographic-narrowness disclosures per writing_guide §10.1 and §10.11, with patient-cluster bootstrap as the primary CI unit on AUROC / slope / Brier skill, the headache-vs-migraine target-asymmetry sentence in the Conclusion, and a decision-curve net-benefit summary line in Results so the TRIPOD+AI coverage map at the foot of the file accurately reflects what the Results paragraph reports).

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
