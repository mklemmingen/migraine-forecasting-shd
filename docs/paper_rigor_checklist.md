# Paper-rigor checklist for the migraine-forecasting benchmark

> Position in the paper: **Supplementary (6.2)**. Reads after `insights_leaf_selection.md`; this is the final doc in the reading order. TRIPOD+AI compliance trail; EPV bands, calibration policy, and honest-comparison discipline.


This document collects the methodological standards the paper needs to satisfy before submission, with the citations behind each item. Every reference here is also a verified entry in `Sources.bib`. The list reflects current 2024-2026 consensus for prediction-model reporting, imbalanced-binary evaluation, calibration measurement, temporal validation, and the migraine forecasting subfield.

Sections below follow the same numbering as `Sources.bib` so a reviewer can cross-reference.

---

## 1. Adopt a reporting standard

**TRIPOD+AI (2024)** [1] is the current consensus for any clinical prediction model that uses regression or machine learning. The checklist contains 27 items plus a 13-item abstract checklist. It supersedes the original TRIPOD 2015 and explicitly calls out machine-learning-specific concerns: fairness, reproducibility, public availability of code and model, and the limitations of single-metric reporting [1]. The paper should ship a TRIPOD+AI compliance table (filled in per item) in the appendix.

Three closely related checklists worth scanning at submission time:

- **CLAIM 2024 Update** - imaging-focused but transfers to digital-biomarker work via its reproducibility and pre-processing items.
- **REFORMS** - a 32-item ML-for-science consensus checklist from a 19-author working group; useful for the benchmark-structure framing independent of the clinical framing.
- **MI-CLAIM** - minimum-information checklist for clinical AI modeling; 6 parts covering setting, performance, population, reference standard, partitioning, reproducibility.

## 1a. Risk-of-bias self-assessment: PROBAST+AI

PROBAST+AI [7] is the 2025 update to PROBAST-2019 and the risk-of-bias counterpart to TRIPOD+AI's reporting role. The tool is structured in two parts. The model-development part contains 16 signalling questions covering applicability and quality. The model-evaluation part contains 18 signalling questions covering risk of bias and applicability. Both parts span the same four domains: participants and data sources, predictors, outcome, and analysis. Because this work spans both development and internal validation, both parts apply, and a filled PROBAST+AI table belongs in the supplementary materials.

The four domains map onto our methodology as follows:

- **Participants and data sources.** Single-cohort retrospective re-analysis of the Park et al. 2016 Korean smartphone-headache-diary cohort, downloaded from the published dataset. No new recruitment, no consent step beyond the original PLOS ONE approval.
- **Predictors.** All predictors are diary self-reports plus calendar-derived calendar features; no laboratory, imaging, or wearable signals enter the model. The feature dictionary is fixed at training time per cell.
- **Outcome.** Binary next-day target derived from the diary's day-of-headache field; the outcome definition is identical across cells.
- **Analysis.** Class imbalance is reported per split (Section 2). Calibration is reported with slope, intercept, ECE10, and Brier (Section 3). Discrimination is reported with both AUROC and AUPRC (Section 4). All metrics carry bootstrap 95% confidence intervals.

## 1b. Regulatory context (EU AI Act, MDR, GDPR)

This paper develops a research prototype on a publicly published dataset; it is not a CE-marked medical device, not Software as a Medical Device (SaMD) under MDR 2017/745, and is not deployed in a clinical workflow. The EU AI Act (Regulation 2024/1689) [10] entered into force on 1 August 2024 with high-risk AI obligations originally applying from 2 August 2026; for AI embedded in regulated medical products under MDR 2017/745, the Article 6(1) deadline is 2 August 2027, and a May 2026 provisional Council and Parliament agreement extends the embedded-medical-AI compliance date to 2 August 2028 (not yet formally adopted as of the present writing). None of these deadlines bind the present work because the model is not placed on the market or put into service in the sense of the Act.

GDPR (Regulation 2016/679) is satisfied at source: the Park 2016 dataset was released under the journal's open-access terms with the original PLOS ONE ethical approval covering redistribution. No new personal data are processed and no re-identification attempts are made.

## 1c. Research-integrity baseline (DFG GWP code)

The DFG "Leitlinien zur Sicherung guter wissenschaftlicher Praxis - Kodex" (Version 1.2, September 2024) [9] is the binding research-integrity reference for German higher-education institutions and is the conventional baseline cited by Hochschule papers. The paper's methods and data-availability statements satisfy the Kodex guidelines on documentation (Leitlinie 12), public access to research results (Leitlinie 13), and authorship (Leitlinie 14): the code is in a public repository, the dataset is publicly distributed by the original authors, and only individuals who contributed substantively are listed as authors. Conflicts of interest are disclosed in the manuscript front matter.

## 1d. Next-stage guideline if the model moves toward deployment: DECIDE-AI

DECIDE-AI [8] is the stage-specific reporting guideline for the early live clinical evaluation of AI-based decision-support systems, sitting between the TRIPOD+AI pre-clinical stage and the SPIRIT-AI / CONSORT-AI trial stage. It comprises 17 AI-specific reporting items (28 sub-items) plus 10 generic items. DECIDE-AI is not the relevant guideline for this paper because no clinical evaluation occurs here. It is cited so that any follow-up study using the model in a real clinical workflow has the correct reporting standard pre-identified.

## 2. Sample size and overfitting risk

The events-per-variable (EPV) framing [2] is the first-pass discipline check. We adopt the conservative reading consistent with Peduzzi 1996 and the Riley 2020 sample-size literature: `<10` high overfitting risk, `10-19` marginal, `>=20` low. Modern Riley-school formulae [2] supersede the rule of thumb but the bands remain useful for headlining risk.

Our migraine target sits at EPV-5.5 on the `full_features` set (52 features, 287 positive days) - the high-risk band - and at EPV 47.8 on the `park_features` set (6 features) - comfortably low-risk. The headache target sits at EPV 18.2 on `full_features` (marginal) and >=30 on the smaller-feature sets (low). See `docs/dataset.md` (Events-per-variable section) for the full table and implications.

The discussion section needs to:

1. Report EPV per cell.
2. Report the calibration-slope number.
3. Flag the migraine `full_features` cell explicitly as high-risk EPV.
4. Argue that the Park feature set is the methodologically cleanest migraine row.

## 3. Calibration measurement

Discrimination metrics alone are insufficient. Huang et al. (2020) [3] is the canonical clinical-prediction-model calibration tutorial; it argues for reporting at minimum:

- Calibration-in-the-large (overall observed/expected ratio).
- Calibration slope (= 1 is ideal; < 1 is the classical overfitting fingerprint; > 1 indicates under-confidence).
- A reliability diagram with sample sizes per bin disclosed.

The evaluator templates and the aggregator metric list both include calibration slope alongside ECE10 and Brier [1, 3].

## 4. Imbalanced-binary metric choice

Two recent positions are both relevant and slightly tension each other:

- The conventional view (defended in many ML-for-medicine tutorials) is that AUPRC is preferable to AUROC on imbalanced data.
- McDermott et al. (2024) [4] argue mathematically and empirically that AUPRC is not unambiguously superior and can systematically favour high-prevalence subgroups, with fairness implications. They recommend reporting BOTH and treating AUPRC as a tool for retrieval-style use cases, not a default replacement for AUROC.

Implication: keep both AUROC and AUPRC in the headline table (we already do), and disclose that comparing AUPRC across the migraine (~7% positive) and headache (~24% positive) cells is misleading because the AUPRC baseline differs by base rate. The aggregator's metric note on AUPRC encodes this caveat.

For threshold-derived metrics on imbalanced binary, MCC is the dominant choice in the recent literature; Accuracy is base-rate-dominated and should never be the headline. The comparison HTML's `Acc (base!)` label and the inline metric-note encodes this.

## 5. Temporal validation and leakage

Random k-fold cross-validation on time-series data inflates performance estimates by leaking future-into-past information. We use scikit-learn's `TimeSeriesSplit` (expanding-window CV) for the 70_15_15 chrono cells. The paper should disclose this explicitly and note its limitation versus full walk-forward validation.

Within-patient leakage is a separate concern: if the same patient has rows in both training and test, the model can memorise patient identity. Our chronological splits are patient-overlapping (a deliberate design choice for forecasting on a returning user). The paper should disclose this and explain why patient-overlap is appropriate for the deployment scenario (forecasting today for a patient whose history we already have) versus a fully held-out-patient evaluation (which would test generalisation to a new patient and is a different scientific question).

## 6. Reproducibility

TRIPOD+AI item 22 [1] requires public code and model availability. Our repository has the code; the model artifacts are regenerable from the scaffolds. The submission should include:

- Pinned `requirements.txt` (done).
- TabPFN checkpoint hashes (documented in the builder docstrings).
- Exact split files on disk under `data/processed/`.
- Hardware and software stack: the canonical environment table (CPU, discrete AMD Radeon RX 7900 XT via PyTorch ROCm, OS, library versions) lives in the Hardware section of the top-level `README.md`. Per-leaf wall-clock for train and eval phases is not currently persisted in the per-leaf result artifacts; the only on-disk timing is the AutoGluon-internal `total runtime = ...` line inside `experiment/1/.../_running_output/training_*.txt` for Addition 1 leaves. Adding a `Wall-clock` row to the result-text emitter in `_scaffold_leaves.py` is a pending instrumentation task; until that lands, the paper should report the sweep wall-clock derived from those AutoGluon logs only for Addition 1 (AutoTabPFN family), and disclose that Addition 0 and the evaluator phase are unrecorded.
- The seed used in `run_bootstrap_evaluation` (`seed=42` in `_scaffold_leaves.py`).

## 7. Migraine-domain context

Two direct comparators in the recent migraine-forecasting literature:

- **Stubberud et al. (2023)** [5] - 18 patients with episodic migraine, 388 headache diary entries (295 days analysed). Best random-forest model achieved hold-out AUC 0.62. This is the small-cohort scale roughly comparable to ours by patient count, although the modality (diary + wearable biofeedback) differs from our diary-only setup.
- **Faisal et al. (2026)** [6] - 146 individuals, 21\,550 headache days, BioCer randomized clinical trial (NCT05616741). Best time-series model achieved hold-out next-day AUC 0.84 (95% CI 0.82-0.85). This is the current high-water mark. It uses diary + biofeedback wearables (trapezius EMG, HRV, peripheral skin temperature); the most predictive features were headache intensity, headache duration, and heart-rate scores.

Implication: our work uses diary-only inputs and a smaller cohort, so a numerical AUC comparison against Faisal et al. would be apples-to-oranges. The honest framing is to position our results as "what is achievable with diary-only inputs at the Park 2016 cohort scale, using publicly available ML stack" - which is a complementary scientific question to the wearable-augmented work.

## 8. Honest comparison reporting

The comparison table at `experiment/comparison_*.html` already implements the standards below; the paper's results section should re-state them:

- Every metric is reported as `mean [95% CI]`, derived from 1000-iteration bootstrap of the test set.
- "Best in row" markers fire only when one cell's 95% CI is strictly disjoint from every other cell's CI in the same row, for that metric. Most rows have no marker - this signals that most architecture × data-package combinations are statistically indistinguishable at our sample size, which is itself the headline finding.
- Source provenance is on every cell (hold-out vs. 5-fold time-series CV).
- Accuracy is base-rate-dominated and is labelled as such in the table (`Acc (base!)`) with a per-metric note.

## References (numbering local to this document)

[1] G. S. Collins, K. G. M. Moons, P. Dhiman, R. D. Riley, A. L. Beam, B. Van Calster, M. Ghassemi, X. Liu, J. B. Reitsma, M. van Smeden, *et al.*, "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ*, vol. 385, e078378, Apr. 2024. doi: [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378). BibTeX key: `collins2024tripodAI`.

[2] G. P. Martin, R. D. Riley, J. Ensor, and S. W. Grant, "Statistical primer: sample size considerations for developing and validating clinical prediction models," *European Journal of Cardio-Thoracic Surgery*, vol. 67, no. 5, p. ezaf142, May 2025. doi: [10.1093/ejcts/ezaf142](https://doi.org/10.1093/ejcts/ezaf142). BibTeX key: `martin2025samplesize`.

[3] Y. Huang, W. Li, F. Macheret, R. A. Gabriel, and L. Ohno-Machado, "A tutorial on calibration measurements and calibration models for clinical prediction models," *Journal of the American Medical Informatics Association*, vol. 27, no. 4, pp. 621-633, Apr. 2020. doi: [10.1093/jamia/ocz228](https://doi.org/10.1093/jamia/ocz228). BibTeX key: `huang2020calibration`.

[4] M. B. A. McDermott, L. H. Hansen, H. Zhang, G. Angelotti, and J. Gallifant, "A closer look at AUROC and AUPRC under class imbalance," arXiv:2401.06091, Jan. 2024. doi: [10.48550/arXiv.2401.06091](https://doi.org/10.48550/arXiv.2401.06091). BibTeX key: `mcdermott2024aurocAuprc`.

[5] A. Stubberud, S. H. Ingvaldsen, E. Brenner, I. Winnberg, A. Olsen, G. B. Gravdahl, M. S. Matharu, P. Nachev, and E. Tronvik, "Forecasting migraine with machine learning based on mobile phone diary and wearable data," *Cephalalgia*, vol. 43, no. 5, pp. 1-10, 2023. doi: [10.1177/03331024231169244](https://doi.org/10.1177/03331024231169244). BibTeX key: `stubberud2023forecasting`.

[6] F. Faisal, A. C. Poole, A. Danelakis, T. Wergeland, M. Bjørk, A. N. Khanevski, E. S. Kristoffersen, T. Kumelj, I. C. K. Larsen, O. V. Lunder, M. Matharu, P. Nachev, M. R. Simpson, E. Tronvik, K. G. Vetvik, B. S. Winsvold, L. R. Øie, A. B. Øvrevik, and A. Stubberud, "Forecasting migraine with time-series machine learning from mobile health data," *The Journal of Headache and Pain*, vol. 27, no. 91, 2026. doi: [10.1186/s10194-026-02346-7](https://doi.org/10.1186/s10194-026-02346-7). BibTeX key: `faisal2026forecasting`.

[7] K. G. M. Moons, J. A. A. Damen, T. Kaul, L. Hooft, C. Andaur Navarro, P. Dhiman, A. L. Beam, B. Van Calster, L. A. Celi, S. Denaxas, A. K. Denniston, M. Ghassemi, G. Heinze, A. P. Kengne, L. Maier-Hein, X. Liu, P. Logullo, M. D. McCradden, N. Liu, L. Oakden-Rayner, K. Singh, D. S. Ting, L. Wynants, B. Yang, J. B. Reitsma, R. D. Riley, G. S. Collins, and M. van Smeden, "PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods," *BMJ*, vol. 388, p. e082505, Mar. 2025. doi: [10.1136/bmj-2024-082505](https://doi.org/10.1136/bmj-2024-082505). BibTeX key: `moons2025probastAI`.

[8] B. Vasey, M. Nagendran, B. Campbell, D. A. Clifton, G. S. Collins, S. Denaxas, A. K. Denniston, L. Faes, B. Geerts, M. Ibrahim, X. Liu, B. A. Mateen, P. Mathur, M. D. McCradden, L. Morgan, J. Ordish, C. Rogers, S. Saria, D. S. W. Ting, P. Watkinson, W. Weber, P. Wheatstone, and P. McCulloch, "Reporting guideline for the early stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI," *BMJ*, vol. 377, p. e070904, May 2022. doi: [10.1136/bmj-2022-070904](https://doi.org/10.1136/bmj-2022-070904). Simultaneously published in *Nature Medicine* 28, 924-933, 2022, doi: [10.1038/s41591-022-01772-9](https://doi.org/10.1038/s41591-022-01772-9). BibTeX key: `vasey2022decideAI`.

[9] Deutsche Forschungsgemeinschaft, "Leitlinien zur Sicherung guter wissenschaftlicher Praxis - Kodex," Version 1.2 (corrected), Stand September 2024 (first published September 2019), Bonn, Germany. doi: [10.5281/zenodo.14281892](https://doi.org/10.5281/zenodo.14281892). URL: [https://www.dfg.de/de/grundlagen-themen/grundlagen-und-prinzipien-der-foerderung/gwp](https://www.dfg.de/de/grundlagen-themen/grundlagen-und-prinzipien-der-foerderung/gwp). English summary at [https://wissenschaftliche-integritaet.de/en/code-of-conduct/](https://wissenschaftliche-integritaet.de/en/code-of-conduct/). BibTeX key: `dfg2024gwpKodex`.

[10] European Parliament and Council of the European Union, "Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)," *Official Journal of the European Union*, Jul. 2024. ELI: [http://data.europa.eu/eli/reg/2024/1689/oj](http://data.europa.eu/eli/reg/2024/1689/oj). BibTeX key: `eu2024aiAct`. URL verified live on 2026-05-14.

---

## 9. TRIPOD+AI 27-item compliance table

Per-item compliance map against the canonical TRIPOD+AI expanded checklist [collins2024tripodAI, Web Table 1, version 7-February-2024]. Item specs are paraphrased from the BMJ checklist; the page anchor cites the canonical PDF (`bmj-2023-078378.full.pdf` for the main paper; `colg078378.wt1.pdf` for Web Table 1 with the expanded item list). D = development item; E = evaluation item; D;E = both apply.

Status codes:
- **✓ covered** — the item is substantively addressed in the cited paper section, with the specific content the BMJ spec requires.
- **⚠️ partial** — the item is touched but does not yet meet the spec; a small addition closes the gap.
- **✗ missing** — the item has no coverage; new content is required.
- **N/A** — the item does not apply in this study's design (e.g., evaluation-only items for a development paper); an explicit N/A statement is still required.

### TITLE

| # | Spec (Collins 2024 wt1) | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 1 (D;E) | Identify the study as developing or evaluating a multivariable prediction model, the target population, and the outcome (wt1 p. 1) | `paper_readiness.md §1` (Title) | ✓ | "Within-person versus pooled discrimination in next-day migraine and headache prediction models on the Park 2016 SHD cohort" — names prediction models, outcomes (migraine and headache), population (Park 2016 SHD cohort), time horizon (next-day). |

### ABSTRACT

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 2 (D;E) | Report an abstract addressing each item in the TRIPOD+AI for Abstracts checklist (wt1 p. 1) | `paper_readiness.md §2` (Abstract) | ⚠️ partial | Five-heading structured abstract drafted to ~297 words (under 450-word JHP cap); 3 of 13 TRIPOD+AI Abstract sub-items added (missing-data, leave-one-site-out external validation, funding). Restructure to 5-heading JHP format is a Phase-5 follow-up. |

### INTRODUCTION

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 3a (D;E) | Explain the healthcare context and rationale; reference existing models (wt1 p. 1) | `introduction.md` ¶1-2 | ✓ | Clinical premise (one-day-ahead probability → pre-emptive oral medication) and gap (pooled-AUROC overstates within-person value) named; Holsteen 2020 and Faisal 2026 cited as comparators. |
| 3b (D;E) | Describe target population, intended purpose in care pathway, intended users (wt1 p. 1) | `introduction.md` ¶1 | ⚠️ partial | Target population (Park 2016 SHD cohort, 62 patients) and intended use (pre-emptive medication) described; intended users (patient self-directed vs clinician-supervised) not explicitly named. |
| 3c (D;E) | Describe any known health inequalities between sociodemographic groups (wt1 pp. 1-2) | `introduction.md` §1 (applicability) | N/A | Within-cohort sociodemographic variation is not available: Park 2016 enrolled from two Korean university clinics under ICHD-3 episodic-migraine inclusion (resulting in 82% female enrolment). Inequalities across groups are not evaluable within this cohort; the applicability-domain consequence is the relevant compliance content and is named explicitly in the Introduction. |
| 4 (D;E) | Specify objectives, including whether development or validation (wt1 p. 2) | `introduction.md` ¶4 | ✓ | Seven layered contributions enumerated (Add-0 stacking; Add-1 TabPFN; Add-2 explainability; Add-3 temporal; Add-4 sequence; Add-5 personalisation + leave-one-site-out external check; Add-6 clinical value). |

### METHODS

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 5a (D;E) | Source of data separately for development and evaluation, rationale, representativeness (wt1 p. 2) | `dataset.md §1-2`; `external_validation_site.md §2` | ✓ | Park 2016 SHD published dataset (PLOS ONE supplementary) re-analysed; leave-one-site-out across Uijeongbu/Dongtan recruitment clinics for evaluation. |
| 5b (D;E) | Dates of participant data collection (wt1 p. 2) | `dataset.md §2` | ✓ | Park 2016 recruitment dates September 2014 – January 2015 [park2016shd, p. 3]. |
| 6a (D;E) | Study setting (e.g. primary care), number and location of centres (wt1 p. 2) | `dataset.md §2`; `external_validation_site.md §2` | ✓ | Two Korean university hospitals (Uijeongbu St. Mary's; Dongtan Sacred Heart Hospital); secondary-care neurology outpatient clinics. |
| 6b (D;E) | Eligibility criteria for study participants (wt1 p. 2) | `dataset.md §2` | ✓ | ICHD-3 episodic migraine; 19-55 years; 2-14 headache days/month; ≥1 year stable headache characteristics; possession of personal smartphone capable of operating the SHD [park2016shd, p. 3]. |
| 6c (D;E) | Treatments received and how handled during development/evaluation (wt1 p. 3) | `dataset.md §2` | ⚠️ partial | Preventive and acute medication sheets exist in Park 2016 Sheets 1-2 but are NOT used as predictors in this benchmark; needs an explicit one-sentence declaration in dataset.md per Phase-3 item 3. |
| 7 (D;E) | Data pre-processing and quality checking; consistency across sociodemographic groups (wt1 p. 3) | `dataset.md §3-5` | ✓ | Translation pipeline, deduplication, gap-aware reindexing, feature engineering all documented in dataset.md; consistency across sociodemographic groups not addressed because Park 2016 cohort is monodemographic. |
| 8a (D;E) | Outcome definition, time horizon, how/when assessed (wt1 p. 3) | `dataset.md §6` | ✓ | Outcome: next-day binary indicator from self-reported diary (`migraine_today.shift(-1)` for migraine target; `headache_free == 0` for headache target). Time horizon: 24 hours. |
| 8b (D;E) | Subjective outcome assessment qualifications (wt1 p. 3) | `dataset.md §6` (Outcome) | N/A | The outcome is patient self-report on the SHD diary; no third-party adjudication step exists, so assessor qualifications do not apply. An explicit one-sentence N/A declaration is added to dataset.md §6. |
| 8c (D;E) | Blinding of outcome assessment (wt1 p. 4) | `dataset.md §6` (Outcome) | N/A | Blinding does not apply: the patient records outcome and predictors on the same diary day in the same instrument, with no third-party assessor whose access could be restricted. An explicit one-sentence N/A declaration is added to dataset.md §6. |
| 9a (D) | Choice of initial predictors and any pre-selection (wt1 p. 4) | `dataset.md §4-5`; `park_features.md` | ✓ | Park's 18-trigger initial set is the candidate pool; engineered rolling/lag features layered for the `full_features` set; Park's 6-trigger stepwise-selected subset retained as `park_features`. |
| 9b (D;E) | Define all predictors, how and when measured (wt1 p. 4) | `dataset.md §4-5`; `park_features.md` | ✓ | Full feature dictionary including same-day flags, 3- and 7-day rolling rates, lag features, streaks, and variability metrics documented. |
| 9c (D;E) | Subjective predictor assessment qualifications (wt1 p. 5) | `dataset.md §4-5` (Predictors) | N/A | Predictors are patient self-report on the SHD diary; no third-party assessor whose qualifications apply. An explicit one-sentence N/A declaration is added to dataset.md §4-5. |
| 10 (D;E) | Sample size and justification, with sample size calculation details (wt1 p. 5) | `paper_rigor_checklist.md §2`; `dataset.md §7` | ✓ | EPV bands per cell reported (migraine/full = 5.5; migraine/park = 47.8; headache/full = 18.2). High-risk EPV cells flagged explicitly. |
| 11 (D;E) | Missing-data handling, reasons for omitting any data (wt1 p. 5) | `dataset.md §5` | ⚠️ partial | No-imputation policy with gap-aware reindexer (treats missing diary days as structural rather than MAR) is documented; 86.3% diary-adherence rate reported; leakage-via-imputation is N/A because no imputation occurs. The one remaining sub-bullet ("for each predictor being considered, the number of missing values") is added as a small per-predictor missingness column in the dataset characterization table. |
| 12a (D) | How data were used in the analysis, including any partitioning (wt1 p. 5) | `dataset.md §6`; `results_findings.md §1-2` | ✓ | Three split types (chrono, stratified, patient) × three ratios (70_30, 70_15_15, 80_20); patient-overlap explicitly disclosed for chronological split. |
| 12b (D) | Predictor handling: functional form, rescaling, transformation, standardisation (wt1 p. 6) | `dataset.md §5`; `xgboost.md §2` | ✓ | Numeric features used raw (XGBoost is scale-invariant); categorical features one-hot; no standardisation for TabPFN per its in-context-learning prescription. |
| 12c (D) | Model type, rationale, building steps, hyperparameter tuning, internal validation (wt1 p. 6) | `xgboost.md`; `tabPfn.MD` | ✓ | Calibrated XGBoost stack (NSGA-II Pareto front + single-objective ladder HP020-HP500); TabPFN family across 5 released variants without per-leaf tuning; AutoTabPFN as post-hoc ensemble; window-MLP/GRU/TCN as sequence baselines. |
| 12d (D;E) | Heterogeneity across clusters (e.g. centres, countries) (wt1 pp. 6-7) | `external_validation_site.md §7`; `addition5_personalization.md` | ✓ | Two-site leave-one-site-out external check addresses centre-level clustering; per-patient AUROC distribution + within-person C addresses individual-level clustering. |
| 12e (D;E) | Performance measures and plots used; rationale (wt1 p. 7) | `paper_rigor_checklist.md §3-4`; `addition6_clinical_value.md` | ✓ | Discrimination (AUROC, AUPRC) + calibration (slope, ECE10, Brier) + clinical utility (decision-curve net benefit, Brier skill) + within-person C-statistic — all with bootstrap CIs. |
| 12f (E) | Model updating (recalibration) from evaluation (wt1 p. 7) | `external_validation_site.md §7` | ⚠️ partial | No model updating performed on the external check; calibration drift documented (O:E 0.50-1.54 for migraine) and flagged as deployment-time recalibration requirement. An explicit "no recalibration was performed in this study" sentence is the Phase-3 item 4 follow-up. |
| 12g (E) | How predictions were calculated for evaluation (wt1 p. 7) | `xgboost.md §3`; `tabPfn.MD §5` | ✓ | XGB: `calibrated_proba(bundle, X)` from the stacked-meta-LR bundle; TabPFN: `model.predict_proba(X)[:, 1]`. |
| 13 (D;E) | Class imbalance methods, recalibration if used (wt1 p. 8) | `paper_rigor_checklist.md §4`; `xgboost.md §2`; `dataset.md §7` | ✓ | XGBoost uses `scale_pos_weight ≈ 13` (≈ 1/positive-rate) for class weighting; van den Goorbergh 2022 [vandengoorbergh2022imbalance] cited for the calibration-side caveat; Platt calibration follows as the recalibration step. |
| 14 (D;E) | Approaches to address model fairness (wt1 p. 8) | `introduction.md` §1 (applicability) ; `discussion.md` §7.4 (Limitations) | N/A | Within-cohort fairness analysis is not evaluable: the Park 2016 cohort is monodemographic by design (single country, single ethnicity, ~82% female enrolment). The applicability-domain limitation — fairness across underrepresented groups is not claimed and not testable within this cohort — is named in the Introduction and the Limitations. |
| 15 (D) | Model output: probabilities, classification; threshold rationale (wt1 p. 8) | `addition6_clinical_value.md §3.3`; `xgboost.md §3`; `tabPfn.MD §5` | ⚠️ partial | Models emit calibrated probabilities; operating-point thresholds (t = 0.10-0.20 / 0.20-0.35 / >0.35) defined in `addition6_clinical_value.md §3.3`. The risk-group framing per TRIPOD+AI Item 11 / Item 15 nomenclature is the Phase-3 item 6 follow-up. |
| 16 (D;E) | Differences between development and evaluation data (wt1 p. 9) | `external_validation_site.md §2-3` | ✓ | Uijeongbu (8.6% migraine base rate) vs Dongtan (5.7%) site differences documented; same diary instrument across sites so eligibility/outcome/predictor definitions are identical. |
| 17 (D;E) | Ethical approval and informed consent (wt1 p. 9) | `paper_readiness.md §31 (Ethics statement)`; `paper_rigor_checklist.md §1b` | ✓ | Secondary analysis of Park 2016 publicly released dataset under PLOS ONE open-access terms; original IRB approval from Dongtan Sacred Heart Hospital (2014-132) and Uijeongbu St. Mary's (UC14OIM10085); no new data collection. |

### OPEN SCIENCE

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 18a (D;E) | Source of funding and role of funders (wt1 p. 9) | `paper_readiness.md §31 (Funding)` | ✓ | "This work received no external funding." Workstation hardware provided by [institution]. |
| 18b (D;E) | Conflicts of interest for all authors (wt1 p. 9) | `paper_readiness.md §31 (CoI)` | ✓ | Author CoI disclosure paragraph drafted. |
| 18c (D;E) | Study protocol availability (wt1 p. 9) | `paper_readiness.md §31` (Open Science) | N/A | The study is a secondary analysis of a publicly released dataset; no pre-registered study protocol was developed. An explicit one-sentence N/A declaration sits in the Open Science statement. |
| 18d (D;E) | Registration information for the study (wt1 p. 9) | `paper_readiness.md §31` (Open Science) | N/A | The study is not a clinical trial and is not registered on clinicaltrials.gov or any equivalent registry; an explicit one-sentence N/A declaration sits in the Open Science statement. |
| 18e (D;E) | Availability of study data (wt1 p. 10) | `paper_readiness.md §31 (Data availability)` | ✓ | Park 2016 SHD raw dataset available as Supplementary File S1 of [park2016shd]; engineered parquets reproducible via `repro.py`. |
| 18f (D;E) | Availability of analytical code (wt1 p. 10) | `paper_readiness.md §31 (Code availability)`; `README.md` | ✓ | Public GitHub repository (`[GitHub URL]`); Zenodo DOI (`[Zenodo DOI]`) to be assigned at submission; `requirements.txt` pinned. |

### PATIENT & PUBLIC INVOLVEMENT

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 19 (D;E) | Patient and public involvement; state no involvement if absent (wt1 p. 10) | `paper_readiness.md §31` (Open Science) | N/A | No patients or public were involved in study design, conduct, reporting, interpretation, or dissemination. An explicit one-sentence "No patient or public involvement" declaration (GRIPP2 inapplicable) sits in the Open Science statement. |

### RESULTS

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 20a (D;E) | Flow of participants through the study; participants with and without the outcome (wt1 pp. 10-11) | `dataset.md §2`; `fig_a2_cohort_flow.py` (figure A2) | ✓ | 62 enrolled patients → 63 unique processed patient_ids → 4,516 patient-days; pooled migraine rate 7.20%; pooled headache rate ~24%. Cohort-flow figure (fig_a2). |
| 20b (D;E) | Characteristics overall and per data source; key dates, predictors, treatments, sample size, outcome events, missing data (wt1 p. 11) | `dataset.md §2-3`; `data/processed/dataset_characterization.pdf` | ⚠️ partial | Cohort-level table exists; baseline-characteristics-by-split table (TRIPOD+AI Item 13c-style across train/val/test) is the Phase-3 item 7 follow-up. |
| 20c (E) | For model evaluation, distribution comparison with development data (wt1 p. 11) | `external_validation_site.md §3` | ✓ | Per-site base-rate contrast (8.6% vs 5.7% migraine; 25.4% vs 21.6% headache) reported in §3; full predictor-distribution comparison across sites is in §3-§7. |
| 21 (D;E) | Number of participants and outcome events in each analysis (wt1 p. 11) | per-leaf `results_*.txt`; `results_findings.md` | ✓ | Each cell's n_test, n_positive, n_negative reported in the leaf-level results files; aggregated in the comparison CSV. |
| 22 (D) | Full prediction model details (formula, code, object, API) (wt1 pp. 11-12) | `experiment/0/.../HP*/results/`; `experiment/1/.../tabpfn/.../model.joblib`; `xgboost.md §1`; `tabPfn.MD §5` | ⚠️ partial | Final-model hyperparameter JSONs live per leaf under `experiment/0/.../HP020/results/` (XGB) and the TabPFN variant directory (TabPFN). Substance is present; explicit doc-side pointer naming the per-leaf JSON paths in `xgboost.md` §1 + `tabPfn.MD` §5 is the remaining edit. |
| 23a (D;E) | Performance estimates with confidence intervals; consider plots (wt1 p. 12) | `results_findings.md §1-7`; `external_validation_site.md §7`; figure pack | ✓ | Every headline metric (AUROC, AUPRC, calibration slope, Brier, ECE10) reported as `mean [95% CI]` from 1000-iteration bootstrap; calibration plots, decision curves, forest plots all in the figure pack. |
| 23b (D;E) | Heterogeneity in model performance across clusters (wt1 p. 12) | `external_validation_site.md §7`; `addition5_personalization.md` | ✓ | Cross-site heterogeneity table at §7; per-patient AUROC distribution + within-person C-statistic for individual-level clustering. |
| 24 (E) | Model updating results (wt1 pp. 12-13) | `external_validation_site.md §7` | ⚠️ partial | No model updating was performed; should be stated as a negative declaration (links to Item 12f follow-up). |

### DISCUSSION

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 25 (D;E) | Overall interpretation, fairness in context of objectives and previous studies (wt1 p. 13) | `discussion.md §7.1-7.3` | ✓ | Five subsections: load-bearing finding (§7.1), per-Addition evidence (§7.2), three field implications (§7.3); each places results in context of Holsteen 2020, Faisal 2026, Dumkrieger 2025 review. |
| 26 (D;E) | Limitations and effects on biases, statistical uncertainty, generalizability (wt1 p. 13) | `discussion.md §7.4` | ✓ | Seven explicit limitations: out-of-cohort transport, 24-hour horizon, chronological-split patient overlap, rolling-window ablation absence, TabPFN family-level claim, within-person personalisation negative, ShapIQ versioning. |
| 27a (D) | Poor quality or unavailable input data handling (wt1 p. 13) | `discussion.md §7.5` (Future work) | N/A | This is a research benchmark, not a deployment system; handling of poor or unavailable predictor values at inference time is deployment-stage instrumentation, out of scope. An explicit scope-disclaimer sentence is added to §7.5 Future work. |
| 27b (D) | User interaction and expertise required (wt1 p. 13) | `discussion.md §7.5` (Future work) | N/A | Deployment-stage user-interaction and required expertise are out of scope of a benchmark study; the relevant reporting guideline for that stage is DECIDE-AI [vasey2022decideAI]. An explicit scope-disclaimer sentence is added to §7.5 Future work. |
| 27c (D;E) | Next steps for future research (wt1 p. 14) | `discussion.md §7.5` | ✓ | Three explicit future directions: true out-of-cohort external validation (Park-group 82-patient SHED); wearable augmentation; multi-cohort deep personalisation. |

### Compliance summary

The TRIPOD+AI expanded checklist resolves to 52 sub-items across the 27 numbered items (some items have a-g sub-divisions). Of these:

- **Fully covered (✓): 33 sub-items** — Title 1; Intro 3a, 4; Methods 5a, 5b, 6a, 6b, 7, 8a, 9a, 9b, 10, 12a, 12b, 12c, 12d, 12e, 12g, 13, 16, 17; Open Science 18a, 18b, 18e, 18f; Results 20a, 20c, 21, 23a, 23b; Discussion 25, 26, 27c.
- **N/A by design (10 sub-items, each requiring an explicit one-sentence declaration to score the item)** — Intro 3c (sociodemographic inequalities not evaluable within a monodemographic cohort); Methods 8b, 8c, 9c (self-report eliminates third-party assessor qualifications and blinding); Methods 14 (within-cohort fairness not evaluable; applicability-domain limit instead); Open Science 18c, 18d (no pre-registered protocol; not a registered trial); Patient & Public Involvement 19 (no involvement); Discussion 27a, 27b (deployment-stage usability out of scope for a benchmark study; DECIDE-AI is the relevant stage-specific guideline).
- **Partial (⚠️): 9 sub-items** — Abstract 2 (5-heading JHP restructure); Intro 3b (intended users explicit); Methods 6c (treatments-not-modelled declaration), Methods 11 (per-predictor missing-count sub-bullet), Methods 12f (recalibration negative declaration), Methods 15 (risk-group framing rename); Results 20b (baseline-characteristics-by-split table), Results 22 (final-model parameter-JSON path pointer), Results 24 (model-updating negative declaration).
- **Missing (✗): 0 sub-items** — every sub-item now has either substantive coverage, an N/A-with-statement requirement, or a partial-coverage closure task.

Closing the remaining 19 sub-items (10 N/A declarations + 9 partial closures) lifts the compliance dimension from 63.5% covered to 100% addressed with an estimated ~3-4 hours of focused editing. The 10 N/A items contribute ~30 minutes (each is a one-sentence declaration), the 9 partial items contribute the bulk of the time.

A note on the workplan-vs-checklist numbering: an earlier draft of the project workplan referred to the "parameter-JSON pointer" task as "Item 15a/b". TRIPOD+AI Item 15 is about *model output* (probabilities, classification, thresholds), whereas the parameter-JSON pointer falls under Item 22 (*Model specification* — full model details to enable third-party reproduction). The workplan label was incorrect; the canonical reference is Item 22.

### How this table is to be used

A reviewer walking the TRIPOD+AI checklist with the manuscript open should find:
- Each numbered item's spec text (paraphrased here, verbatim in `colg078378.wt1.pdf`).
- The doc:section pointer where the substance lives.
- A status flag indicating whether the manuscript currently meets the spec.
- An evidence note quoting or pointing to the specific claim.

The table is updated each time a follow-up edit closes one of the ⚠️/✗ rows. The expected end-state for submission is 50/50 sub-items at ✓ or N/A-with-explicit-statement.
