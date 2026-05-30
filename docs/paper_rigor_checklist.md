# Paper-rigor checklist for the migraine-forecasting benchmark

> Position in the paper: **Supplementary (6.2)**. Reads after `insights_leaf_selection.md`; this is the final doc in the reading order. TRIPOD+AI compliance trail; EPV bands, calibration policy, and honest-comparison discipline.


This document collects the methodological standards the paper needs to satisfy before submission, with the citations behind each item. Every reference here is also a verified entry in `Sources.bib`. The list reflects current 2024-2026 consensus for prediction-model reporting, imbalanced-binary evaluation, calibration measurement, temporal validation, and the migraine forecasting subfield.

Sections below follow the same numbering as `Sources.bib` to support cross-reference.

---

## 1. Adopt a reporting standard

**TRIPOD+AI (2024)** [1] is the current consensus for any clinical prediction model that uses regression or machine learning. The checklist contains 27 items plus a 13-item abstract checklist. It supersedes the original TRIPOD 2015 and explicitly calls out machine-learning-specific concerns: fairness, reproducibility, public availability of code and model, and the limitations of single-metric reporting [1]. The paper should ship a TRIPOD+AI compliance table (filled in per item) in the appendix.

Three closely related checklists worth scanning at submission time:

- **CLAIM 2024 Update**: imaging-focused but transfers to digital-biomarker work via its reproducibility and pre-processing items.
- **REFORMS**: a 32-item ML-for-science consensus checklist from a 19-author working group; useful for the benchmark-structure framing independent of the clinical framing.
- **MI-CLAIM**: minimum-information checklist for clinical AI modeling; 6 parts covering setting, performance, population, reference standard, partitioning, reproducibility.

## 1a. Risk-of-bias self-assessment: PROBAST+AI

PROBAST+AI [7] is the 2025 update to PROBAST-2019 and the risk-of-bias counterpart to TRIPOD+AI's reporting role. The tool is structured in two parts. The model-development part contains 16 signalling questions covering applicability and quality. The model-evaluation part contains 18 signalling questions covering risk of bias and applicability. Both parts span the same four domains: participants and data sources, predictors, outcome, and analysis. Because this work spans both development and internal validation, both parts apply, and a filled PROBAST+AI table belongs in the supplementary materials.

The four domains map onto our methodology as follows:

- **Participants and data sources.** Single-cohort retrospective re-analysis of the Park et al. 2016 Korean smartphone-headache-diary cohort, downloaded from the published dataset. No new recruitment, no consent step beyond the original PLOS ONE approval.
- **Predictors.** All predictors are diary self-reports plus calendar-derived calendar features; no laboratory, imaging, or wearable signals enter the model. The feature dictionary is fixed at training time per cell.
- **Outcome.** Binary next-day target derived from the diary's day-of-headache field; the outcome definition is identical across cells.
- **Analysis.** Class imbalance was reported per split (Section 2). Calibration was reported with slope, intercept, ECE10, and Brier (Section 3). Discrimination was reported with both AUROC and AUPRC (Section 4). All metrics carried bootstrap 95% confidence intervals.

## 1b. Regulatory context (EU AI Act, MDR, GDPR)

This paper develops a research prototype on a publicly published dataset; it is not a CE-marked medical device, not Software as a Medical Device (SaMD) under MDR 2017/745, and is not deployed in a clinical workflow. The EU AI Act (Regulation 2024/1689) [10] entered into force on 1 August 2024 with high-risk AI obligations originally applying from 2 August 2026; for AI embedded in regulated medical products under MDR 2017/745, the Article 6(1) deadline is 2 August 2027, and a May 2026 provisional Council and Parliament agreement extends the embedded-medical-AI compliance date to 2 August 2028 (not yet formally adopted as of the present writing). None of these deadlines bind the present work because the model is not placed on the market or put into service in the sense of the Act.

GDPR (Regulation 2016/679) is satisfied at source: the Park 2016 dataset was released under the journal's open-access terms with the original PLOS ONE ethical approval covering redistribution. No new personal data are processed and no re-identification attempts are made.

## 1c. Research-integrity baseline (DFG GWP code)

The DFG "Leitlinien zur Sicherung guter wissenschaftlicher Praxis - Kodex" (Version 1.2, September 2024) [9] is the binding research-integrity reference for German higher-education institutions and is the conventional baseline cited by Hochschule papers. The paper's methods and data-availability statements satisfy the Kodex guidelines on documentation (Leitlinie 12), public access to research results (Leitlinie 13), and authorship (Leitlinie 14): the code is in a public repository, the dataset is publicly distributed by the original authors, and only individuals who contributed substantively are listed as authors. Conflicts of interest are disclosed in the manuscript front matter.

## 1d. Next-stage guideline if the model moves toward deployment: DECIDE-AI

DECIDE-AI [8] is the stage-specific reporting guideline for the early live clinical evaluation of AI-based decision-support systems, sitting between the TRIPOD+AI pre-clinical stage and the SPIRIT-AI / CONSORT-AI trial stage. It comprises 17 AI-specific reporting items (28 sub-items) plus 10 generic items. DECIDE-AI is not the relevant guideline for this paper because no clinical evaluation occurs here. It is cited so that any follow-up study using the model in a real clinical workflow has the correct reporting standard pre-identified.

## 2. Sample size and overfitting risk

The events-per-variable (EPV) framing [2] is the first-pass discipline check. We adopt the conservative reading consistent with Peduzzi 1996 and the Riley 2020 sample-size literature: `<10` high overfitting risk, `10-19` marginal, `>=20` low. Modern Riley-school formulae [2] supersede the rule of thumb but the bands remain useful for headlining risk.

Our migraine target sat at EPV-5.5 on the `full_features` set (52 features, 287 positive days), the high-risk band, and at EPV 47.8 on the `park_features` set (6 features), comfortably low-risk. The headache target sat at EPV 18.2 on `full_features` (marginal) and >=30 on the smaller-feature sets (low). See `docs/dataset.md` (Events-per-variable section) for the full table and implications.

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
- Hardware and software stack: the canonical environment table (CPU, discrete AMD Radeon RX 7900 XT via PyTorch ROCm, OS, library versions) lives in the Hardware section of the top-level `README.md`. Per-leaf wall-clock for the train phase is recoverable from three on-disk signals, ordered by precision: (a) the explicit `# Generated:` and `# Finished:` ISO-timestamp markers wrapping every `_running_output/training_*.txt` (second resolution, present on every wrapped training run across Addition 0 and Addition 1); (b) the AutoGluon-internal `total runtime = NNN.NNs` line (sub-second, AutoTabPFN only); (c) embedded filename timestamps on every `training_*.txt` and `results_*.txt` (form `<prefix>_YYYYMMDD_HHMMSS_<ms>_<uuid>.txt`), which survive `cp -p` and post-hoc `touch` operations that corrupt mtime. `experiment/extract_wallclock.py` walks all three Additions and reports header-precise, AutoGluon-precise, and filename-span per leaf, plus a `multi_day_gap` flag (set when the filename span exceeds 12 hours, indicating train and eval ran in separate sessions). The current extraction covers 465 leaves: 236 Addition 0 + 229 Addition 1, both header-precise (every wrapped training run; the four AutoTabPFN leaves additionally carry AutoGluon-precise); 0 Addition 4 because Addition 4 emits neither `results/` nor `_running_output/`. Addition 0 header-precise training duration: median 2 s, range 1-372 s, total 4.2 h across 236 leaves (the median is dominated by single-fit NonHP variants; the 500-trial Optuna HP-tuned leaves are the minute-scale tail). Addition 1 header-precise: median 1 s, range 0-3404 s, total 9.4 h across 229 leaves; AutoTabPFN per-leaf 1826-3404 s header-precise (2.3× the AutoGluon-internal `total runtime`, the gap being wrapper-script imports, data load, ensemble selection over the 40+ TabPFN copies, and `model.joblib` serialization). Combined header-precise training wall-clock: 13.6 h. The paper reports these aggregates and discloses that header-precise covers only the train phase (the eval phase is not wrapped by a separate log), that the AutoGluon-precise channel underestimates wall-clock by the wrapper overhead factor, and that Addition 4 sequence leaves are excluded because they were never wrapped. Adding a per-leaf `Wall-clock` row to the result-text emitter (`experiment/0/_scaffold_leaves.py`, `experiment/1/_scaffold_leaves.py`, `experiment/4/_scaffold_leaves.py`: three sibling emitters, one per Addition) remains the path to instrumented eval-phase coverage and to Addition 4 coverage.
- The seed used in `run_bootstrap_evaluation` (`seed=42` in `_scaffold_leaves.py`).

## 7. Migraine-domain context

Two direct comparators in the recent migraine-forecasting literature:

- **Stubberud et al. (2023)** [5]: 18 patients with episodic migraine, 388 headache diary entries (295 days analysed). Best random-forest model achieved hold-out AUC 0.62. This is the small-cohort scale roughly comparable to ours by patient count, although the modality (diary + wearable biofeedback) differs from our diary-only setup.
- **Faisal et al. (2026)** [6]: 146 individuals, 21\,550 headache days, BioCer randomized clinical trial (NCT05616741). Best time-series model achieved hold-out next-day AUC 0.84 (95% CI 0.82-0.85). This is the current high-water mark. It uses diary + biofeedback wearables (trapezius EMG, HRV, peripheral skin temperature); the most predictive features were headache intensity, headache duration, and heart-rate scores.

Implication: this work used diary-only inputs on a smaller cohort, so a numerical AUC comparison against Faisal et al. would be apples-to-oranges. The honest framing is to position our results as "what is achievable with diary-only inputs at the Park 2016 cohort scale, using publicly available ML stack", which is a complementary scientific question to the wearable-augmented work.

## 8. Honest comparison reporting

The comparison table at `experiment/comparison_*.html` already implements the standards below; the paper's results section should re-state them:

- Every metric is reported as `mean [95% CI]`, derived from 1000-iteration bootstrap of the test set.
- "Best in row" markers fire only when one cell's 95% CI is strictly disjoint from every other cell's CI in the same row, for that metric. Most rows have no marker; this signals that most architecture × data-package combinations are statistically indistinguishable at our sample size, which is itself the headline finding.
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

Per-item compliance map against the canonical TRIPOD+AI expanded checklist [collins2024tripodAI, Web Table 1, version 7-February-2024]. Item specs are paraphrased from the BMJ checklist (Collins et al. 2024, BMJ vol. 385 p. e078378, DOI 10.1136/bmj-2023-078378); the page anchors use the form "wt1 p. N" where wt1 refers to Web Table 1 (the article's expanded-checklist supplement) and N is the page number within that supplement. D = development item; E = evaluation item; D;E = both apply.

Status codes:
- **✓ covered**: the item is substantively addressed in the cited paper section, with the specific content the BMJ spec requires.
- **⚠️ partial**: the item is touched but does not yet meet the spec; a small addition closes the gap.
- **✗ missing**: the item has no coverage; new content is required.
- **N/A**: the item does not apply in this study's design (e.g., evaluation-only items for a development paper); an explicit N/A statement is still required.

### TITLE

| # | Spec (Collins 2024 wt1) | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 1 (D;E) | Identify the study as developing or evaluating a multivariable prediction model, the target population, and the outcome (wt1 p. 1) | `abstract.md` §Title (L14-15); `body.md` H1 (L1) | ✓ | "Within-person versus pooled discrimination in next-day migraine and headache prediction models on the Park 2016 Korean SHD cohort": names prediction models, outcomes (migraine and headache), population (Korean SHD cohort, Park 2016 corpus), time horizon (next-day). The single demographic anchor "Korean" in the cohort name satisfies the verbatim TRIPOD+AI Item 1 "target population" requirement at the title-page surface where PubMed-scanning neurologists triage; age band (19-55 y), sex distribution (82.3% female), and clinic count (two participating Korean clinics) live in abstract Methods + body §2.1 per TRIPOD+AI Item 8 representativeness placement. |

### ABSTRACT

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 2 (D;E) | Report an abstract addressing each item in the TRIPOD+AI for Abstracts checklist (wt1 p. 1) | `abstract.md` (manuscript front-matter, carried forward into the submission template) | ✓ | 4-heading JHP-structured abstract (Background / Methods / Results / Conclusion), 449 words under the 450-word JHP cap. All 13 TRIPOD+AI for Abstracts sub-items covered (Title, Objective, Setting, Participants, Sample size, Predictors, Outcomes, Statistical methods, Missing-data handling, Performance measures, External validation, Results, Limitations, Funding); coverage map appended at the foot of `abstract.md`. |

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
| 12e (D;E) | Performance measures and plots used; rationale (wt1 p. 7) | `paper_rigor_checklist.md §3-4`; `addition6_clinical_value.md` | ✓ | Discrimination (AUROC, AUPRC) + calibration (slope, ECE10, Brier) + clinical utility (decision-curve net benefit, Brier skill) + within-person C-statistic, all with bootstrap CIs. |
| 12f (E) | Model updating (recalibration) from evaluation (wt1 p. 7) | `external_validation_site.md §7` | ⚠️ partial | No model updating performed on the external check; calibration drift documented (O:E 0.50-1.54 for migraine) and flagged as deployment-time recalibration requirement. An explicit "no recalibration was performed in this study" sentence is the Phase-3 item 4 follow-up. |
| 12g (E) | How predictions were calculated for evaluation (wt1 p. 7) | `xgboost.md §3`; `tabPfn.MD §5` | ✓ | XGB: `calibrated_proba(bundle, X)` from the stacked-meta-LR bundle; TabPFN: `model.predict_proba(X)[:, 1]`. |
| 13 (D;E) | Class imbalance methods, recalibration if used (wt1 p. 8) | `paper_rigor_checklist.md §4`; `xgboost.md §2`; `dataset.md §7` | ✓ | XGBoost uses `scale_pos_weight ≈ 13` (≈ 1/positive-rate) for class weighting; van den Goorbergh 2022 [vandengoorbergh2022imbalance] cited for the calibration-side caveat; Platt calibration follows as the recalibration step. |
| 14 (D;E) | Approaches to address model fairness (wt1 p. 8) | `introduction.md` §1 (applicability) ; `discussion.md` §7.4 (Limitations) | N/A | Within-cohort fairness analysis is not evaluable: the Park 2016 cohort is monodemographic by design (single country, single ethnicity, ~82% female enrolment). The applicability-domain limitation (fairness across underrepresented groups is not claimed and not testable within this cohort) is named in the Introduction and the Limitations. |
| 15 (D) | Model output: probabilities, classification; threshold rationale (wt1 p. 8) | `addition6_clinical_value.md §3.3`; `xgboost.md §3`; `tabPfn.MD §5` | ⚠️ partial | Models emit calibrated probabilities; operating-point thresholds (t = 0.10-0.20 / 0.20-0.35 / >0.35) defined in `addition6_clinical_value.md §3.3`. The risk-group framing per TRIPOD+AI Item 11 / Item 15 nomenclature is the Phase-3 item 6 follow-up. |
| 16 (D;E) | Differences between development and evaluation data (wt1 p. 9) | `external_validation_site.md §2-3` | ✓ | Uijeongbu (8.6% migraine base rate) vs Dongtan (5.7%) site differences documented; same diary instrument across sites so eligibility/outcome/predictor definitions are identical. |
| 17 (D;E) | Ethical approval and informed consent (wt1 p. 9) | `paper_rigor_checklist.md §10 (Ethics statement)`; §1b | ✓ | Secondary analysis of Park 2016 publicly released dataset under PLOS ONE CC BY 4.0 licence; original IRB approval from Dongtan Sacred Heart Hospital (2014-132) and Uijeongbu St. Mary's (UC14OIM10085); no new data collection and no personal communication with the original authors required. |

### OPEN SCIENCE

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 18a (D;E) | Source of funding and role of funders (wt1 p. 9) | `paper_rigor_checklist.md §10 (Funding)` | ✓ | "This work received no external funding." Workstation hardware provided by the author's institution. |
| 18b (D;E) | Conflicts of interest for all authors (wt1 p. 9) | `paper_rigor_checklist.md §10 (Conflicts of interest)` | ✓ | Author conflict-of-interest disclosure paragraph in §10. |
| 18c (D;E) | Study protocol availability (wt1 p. 9) | `paper_rigor_checklist.md §10 (Study protocol)` | N/A | The study is a secondary analysis of a publicly released dataset; no pre-registered study protocol was developed. An explicit one-sentence N/A declaration sits in §10. |
| 18d (D;E) | Registration information for the study (wt1 p. 9) | `paper_rigor_checklist.md §10 (Study registration)` | N/A | The study is not a clinical trial and is not registered on clinicaltrials.gov or any equivalent registry; an explicit one-sentence N/A declaration sits in §10. |
| 18e (D;E) | Availability of study data (wt1 p. 10) | `paper_rigor_checklist.md §10 (Data availability)` | ✓ | Park 2016 SHD raw dataset available as Supplementary File S1 of [park2016shd] under CC BY 4.0; engineered parquets reproducible via `repro.py`. |
| 18f (D;E) | Availability of analytical code (wt1 p. 10) | `paper_rigor_checklist.md §10 (Code availability)`; `README.md` | ✓ | Public GitHub repository (`[GitHub URL]`); Zenodo DOI (`[Zenodo DOI]`) to be assigned at submission; `requirements.txt` pinned. |

### PATIENT & PUBLIC INVOLVEMENT

| # | Spec | Doc:Section | Status | Evidence / note |
|---|---|---|---|---|
| 19 (D;E) | Patient and public involvement; state no involvement if absent (wt1 p. 10) | `paper_rigor_checklist.md §10 (Patient and public involvement)` | N/A | No patients or public were involved in study design, conduct, reporting, interpretation, or dissemination. An explicit one-sentence "No patient or public involvement" declaration (GRIPP2 inapplicable) sits in §10. |

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

The TRIPOD+AI expanded checklist resolves to 52 sub-items across the 27 numbered items. The per-row status flags in the table above record the *pre-closure diagnostic state* of each item; the closures executed on 2026-05-28 are summarised here.

**Pre-closure state (the table above):**

- Fully covered (✓): 33 sub-items.
- N/A by design (10 sub-items): Intro 3c, 14; Methods 8b, 8c, 9c; Open Science 18c, 18d; Patient & Public Involvement 19; Discussion 27a, 27b.
- Partial (⚠️): 9 sub-items: Abstract 2; Intro 3b; Methods 6c, 11, 12f, 15; Results 20b, 22, 24.
- Missing (✗): 0 sub-items.

**Post-closure state (after the 2026-05-28 closure pass):**

| Item(s) | Closure landed in | Result |
|---|---|---|
| 3b, 3c, 14 (applicability + intended users + fairness) | `introduction.md` §1 (new applicability paragraph) | ✓ |
| 6c, 8b, 8c, 9c, 11 (treatments-not-modelled, self-report N/A, per-predictor missing count) | `dataset.md` (new subsections after Study Design and Gap Awareness) | ✓ |
| 12f, 24 (no recalibration, no model updating) | `external_validation_site.md` §7 (new paragraph) | ✓ |
| 15 (risk-group framing rename) | `addition6_clinical_value.md` §3.3 (header rename + spec pointer) | ✓ |
| 18c, 18d, 19 + Open Science end-matter | `paper_rigor_checklist.md` §10 (new section) | ✓ |
| 20b (per-data-source characteristics) | `dataset.md` (new Per-site characteristics table) | ✓ |
| 22 (final-model parameter-JSON path pointer) | `xgboost.md` §1 + `tabPfn.MD` §5 (new subsections) | ✓ |
| 27a, 27b (deployment-stage scope-disclaimers) | `discussion.md` §7.5 (new paragraph) | ✓ |
| **2** (Abstract 4-heading JHP restructure) | `abstract.md` (manuscript front-matter; Background / Methods / Results / Conclusion, 449 words) | ✓ |

**Post-closure totals**: 52 sub-items addressed (33 pre-existing ✓ + 19 closed across this pass: 9 substantive closures and 10 N/A-with-declaration), 0 sub-items remaining ⚠️ or ✗. Compliance is at **100% addressed**, up from 63.5% pre-closure. The 9th substantive closure (Item 2 abstract restructure) was completed by lifting the structured abstract into the public `abstract.md` manuscript-template file with the 4-heading JHP layout (Background / Methods / Results / Conclusion) and the missing TRIPOD+AI Abstract sub-items integrated.

A note on the workplan-vs-checklist numbering: an earlier draft of the project workplan referred to the "parameter-JSON pointer" task as "Item 15a/b". TRIPOD+AI Item 15 is about *model output* (probabilities, classification, thresholds), whereas the parameter-JSON pointer falls under Item 22 (*Model specification*: full model details to enable third-party reproduction). The workplan label was incorrect; the canonical reference is Item 22.

---

## 10. Open Science end-matter declarations

The following declarations close the Open Science items of the TRIPOD+AI checklist (items 18a-f and 19). Each declaration is the one-sentence-or-paragraph statement the checklist requires; when the item is N/A by design (no protocol, no registration, no patient-and-public involvement), the declaration states that fact explicitly with its reason.

### Funding (Item 18a)

This work received no external funding. The hardware used for the experiments (AMD Radeon RX 7900 XT workstation; full specification in the manuscript's Hardware section) was provided by the author's institution. No funder had a role in study design, data analysis, interpretation, or the decision to submit for publication.

### Conflicts of interest (Item 18b)

The authors declare no competing financial or non-financial interests relevant to this work. No financial relationships with any organisation that might have an interest in the submitted work in the past three years; no other relationships or activities that could appear to have influenced the submitted work.

### Study protocol (Item 18c)

No pre-registered study protocol was developed. This study is a secondary analysis of the publicly released Park 2016 Smartphone Headache Diary dataset (Park et al. 2016 [park2016shd]); analysis plans were developed against the published data without registration.

### Study registration (Item 18d)

The study is not a clinical trial and is not registered on clinicaltrials.gov, the EU Clinical Trials Register, the Open Science Framework, or any equivalent registry.

### Data availability (Item 18e)

The Park 2016 SHD raw diary data are publicly available as Supplementary File S1 of [park2016shd], distributed under CC BY 4.0 by PLOS ONE. The engineered split parquet files used in this benchmark are regenerable end-to-end from the published code (`repro.py --what hashes` verifies the 92-parquet content hash log; `repro.py --what aggregate` regenerates the comparison artefacts). The leaf-level model artefacts (`model.joblib`) are regenerable from each leaf's `train.py` script under the deterministic seed `seed=42`. No restrictions apply to retrieval or use of the SHD dataset beyond the CC BY 4.0 attribution requirement.

### Code availability (Item 18f)

The full analytical code, including data engineering, model training, evaluation, and figure rendering, is publicly available at the project repository ([GitHub URL, to be substituted at submission]). A Zenodo snapshot DOI ([Zenodo DOI, to be substituted at submission]) anchors the exact commit corresponding to the submitted manuscript. The Python environment is pinned in `requirements.txt`; the reference build uses Python 3.13.12 on the hardware described above. All packages required to reproduce the reported results in principle are listed in `requirements.txt` with version pins.

### Patient and public involvement (Item 19)

No patients or members of the public were involved in the design, conduct, reporting, interpretation, or dissemination of this study. The original Park 2016 cohort collection (a smartphone diary trial) involved patient participation but the present work is a secondary analysis of the already-published dataset and added no new participant-facing activity. GRIPP2 is therefore not applicable to this study.

### Ethics statement (paired with Item 17)

This study is a secondary analysis of a publicly released dataset; no new participant-facing activity, no new ethics approval, and no contact with the original investigators were required for the present work. The legal basis for re-analysis is the open licence under which the dataset was published, not personal permission from the original authors.

The chain of authorisation is as follows.

1. **Original-study ethics and consent.** The Park 2016 study (PLOS ONE [park2016shd]) obtained IRB approval from Dongtan Sacred Heart Hospital (approval number 2014-132) and Uijeongbu St. Mary's Hospital, Catholic University of Korea College of Medicine (approval number UC14OIM10085), and reports that "the participants received an explanation of the study's aims and procedures and provided written informed consent" [park2016shd, p. 3]. These statements describe the *original* data-collection conditions and are properties of Park et al.'s study, not of the present work.

2. **Public release under CC BY 4.0.** PLOS ONE publishes all articles and their supplementary materials under the Creative Commons Attribution 4.0 International (CC BY 4.0) licence. Park et al. released the diary dataset as Supplementary File S1 of [park2016shd]; the file is therefore distributed under CC BY 4.0, which permits redistribution, reuse, and adaptation provided attribution is given.

3. **The present re-analysis.** This work uses the publicly released Supplementary File S1 under the CC BY 4.0 licence and cites Park et al. 2016 as the source. No written permission from Park et al. was requested or required for this re-analysis, because the licence already covers the use. No re-identification attempts are made; the dataset contains no direct identifiers (it is de-identified by Park et al. at source).

### How this table is to be used

Anyone walking the TRIPOD+AI checklist with the manuscript open should find:
- The doc:section pointer where the substance lives.
- A status flag indicating whether the manuscript currently meets the spec.
- An evidence note quoting or pointing to the specific claim.

The table is updated each time a follow-up edit closes one of the ⚠️/✗ rows. The expected end-state for submission is 50/50 sub-items at ✓ or N/A-with-explicit-statement.


## 11. Reviewer-derived methodological discipline

The following methodological standards were codified after a multi-persona reviewer pass against the JHP submission body. Each standard is grounded in a verified primary source from the CPM literature. They are enforced on every body regen and every long-form doc audit.

### 11.1 CIs on every reported metric (Andaur Navarro 2023)

Every numerical performance figure carries a 95% CI at every appearance in prose: AUROC, AUPRC, calibration slope, calibration intercept, ECE10, Brier score, Brier skill, within-person C-statistic, observed-to-expected ratio. Bootstrap CIs (1,000 iterations) with the resampling unit specified explicitly. Andaur Navarro et al. (J Clin Epidemiol 2023) report that 74.6% of ML-CPM abstracts give discrimination without precision estimates; this is the most prevalent spin pattern in the field.

### 11.2 Active-voice non-deployment disclaimer

Papers without separate-cohort external validation state "we do not recommend clinical deployment" in active voice in (a) abstract Conclusion, (b) Discussion, and (c) Conclusion section. The active form is harder to misquote and prevents the Andaur Navarro 95.2% deployment-overclaim pattern.

### 11.3 Pre-specified, not pre-registered, absent OSF artefact

"Pre-registered" claims require a public registration record (OSF URL, PROSPERO ID, ClinicalTrials.gov entry). Without such an artefact, "pre-specified" or "pre-planned" is used. TRIPOD+AI Item 18d (registration) is a positive declaration; a missing artefact behind a "pre-registered" claim is a spin pattern.

### 11.4 Within-person estimability denominators

Every within-person C-statistic and per-patient AUROC distribution appears with its denominator (X of Y patients estimable at the threshold-event floor). On the present cohort, the migraine within-person finding is evaluable on 19 of 63 patients; this denominator appears in the abstract, body Results, body Discussion, and body Conclusion.

### 11.5 Manuscript licence: CC BY 4.0

Selected at submission, stated explicitly in Declarations. CC BY-NC-ND prohibits LLM analysis of the paper; CC BY permits it. The journal's default is CC BY 4.0; opting in explicitly preserves machine-readable downstream analysis.

### 11.6 Code and data identifiers minted before submission

GitHub URL and Zenodo DOI cited in body Declarations, not left as TBD placeholders. TRIPOD+AI Item 13 requires the identifier.

### 11.7 Riley-school sample-size calculation

Riley / Ensor / Martin framework applied per cell: required N derived from outcome prevalence, target shrinkage (≥ 0.9), expected Cox-Snell R², and target CITL precision (≤ 0.05). On the present cohort, the migraine `full_features` cell (EPV-5.5) sits below the Riley criterion and is declared a priori as an under-powered development cell, not retroactively as a discovered limitation. Peduzzi 1996 EPV is necessary but not sufficient.

### 11.8 Class-imbalance handling: van den Goorbergh chain

Where `scale_pos_weight` or analogous class-weighting is used, the van den Goorbergh et al. 2022 observation is named: reweighting distorts calibration-in-the-large, Platt is the corrective, and the calibration-slope column monitors restoration. The chain (imbalance → reweight → distort CITL → Platt restore → monitor via slope) is explicit in Methods.

### 11.9 Bootstrap resampling unit + paired DeLong assumption

The bootstrap resampling unit (patient-day, patient-cluster, block) is stated in Methods. Patient-day bootstrap under serial dependence underestimates variance; the under-coverage caveat is disclosed when patient-day is used. Paired DeLong (Sun-Xu midrank) assumes within-class independence; on patient-day data the assumption is violated, but Type-I error inflation under the violation runs toward more significance, so null results are conservative against the assumption.

### 11.10 Small-k random-effects pooling

For k < 20 estimable patients, DerSimonian-Laird τ² is biased downward. REML or Paule-Mandel is preferred, or the DL CI is read as a lower bound on between-patient uncertainty.

### 11.11 Demographic narrowness in the abstract Methods

Cohort demographic restrictions (single sex predominance, single ethnicity, narrow age band, single country) are named in the abstract Methods sub-section, not only in body §4.4 Limitations.

### 11.12 No editorial adjectives without numeric anchors

"Substantial", "large", "remarkable", "considerable" - dropped or replaced with the numeric anchor. The reader gets the substantive claim; the adjective is editorial.

### 11.13 Reproduction-plus-extension framing

When a finding reproduces a prior result on a different cohort, the contribution is framed as reproduction-plus-extension: the prior cohort is cited explicitly; the extension half enumerates what the present cohort and methodology add.

### 11.14 Disease-classification version qualifier

Diagnostic-criteria citations name the version including any beta / draft / final qualifier and subset breakdown.

### 11.15 Citation verification before use

Every citation key in body prose resolves in `Sources.bib` and has at least one in-text use. Unused references are dropped or cited.

### Source primary sources for §11

- Andaur Navarro et al. (J Clin Epidemiol 2023), spin patterns in ML CPM: discrimination without CIs (74.6%); deployment overclaim (95.2%).
- Collins et al. (BMJ 2024), TRIPOD+AI 27-item checklist + 13-item Abstract checklist.
- Moons et al. (2025), PROBAST+AI 34 signalling questions across four domains.
- Riley, Ensor, Martin (2020-2025), sample-size framework for binary CPM.
- van den Goorbergh et al. (2022), class-imbalance correction and calibration.
- Huang et al. (2020), calibration measurement (CITL + slope + reliability diagram).
- Hanley and McNeil (1982), C-statistic binormal variance estimator.

The full ruleset (with verbatim primary-source quotes, grouped under banned-word and sentence-structure axes) lives in the project's writing guide and is summarised here for the public-facing record.
