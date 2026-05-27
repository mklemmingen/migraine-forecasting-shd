# Introduction

> Position in the paper: **Introduction (1)**. Synthesises the clinical motivation, the gap in the next-day-diary migraine-forecasting literature, and the contribution.

Migraine is the second-most disabling neurological condition globally, and
its attacks are largely predictable from same-day or recent-day exposures
(stress, sleep disruption, hormonal change, weather, dietary triggers).
The clinical premise that motivates next-day forecasting is that a
sufficiently accurate one-day-ahead probability would let a patient take
an oral pre-emptive medication before symptom onset, converting a reactive
treatment pathway into a forecastable one. The Korean Smartphone Headache
Diary (SHD) cohort of Park et al. 2016 [park2016shd, p. 3] is the largest
open next-day-diary migraine dataset that records the full trigger panel
needed to test that premise: 62 enrolled patients across two neurology
clinics, 4,516 patient-days, daily attack indicator with eighteen
candidate trigger factors recorded each evening.

The next-day-diary migraine-forecasting literature reports discrimination
as a pooled AUROC computed across patient-days. Pooled AUROC mixes two
distinct quantities into a single number: how well the model ranks days
within a patient, and how well it separates high-base-rate patients from
low-base-rate ones. Within-patient time-series methodology in headache
research was first put on an empirical footing by Houle et al. 2005
[houle2005timeseries, pp. 445-446], whose four-week diary study of 49
migraine and tension-type sufferers established that headache on one
day positively autocorrelates with headache the next and argued that
within-patient designs reveal day-to-day patterns that group-level
cross-sectional analyses mask. Holsteen et al. 2020 carried that
within-person framing into a modern forecasting evaluation, reporting
a within-person C-statistic of 0.56 (95% CI 0.54-0.58) on an
independent US cohort while the pooled metric on the same data sat
higher, and argued that within-person discrimination is the fitting
evaluation for an individualised forecast [holsteen2020triggers,
p. 2364]. The
distinction matters clinically: a model whose pooled AUROC reflects
between-patient base-rate separation is useful for triaging
patient-level risk strata, but it cannot tell a given patient which of
*their* days is the attack. The diary-only-forecasting literature
converges in the AUC 0.56-0.73 range under within-person evaluation
[houle2017stress, p. 1041; holsteen2020triggers, p. 2364], well below
the wearable-augmented work that reaches higher numbers on different
inputs (Faisal 2026 reports 0.84 with EMG, HRV, and skin-temperature
signals on 21,550 headache days [faisal2026forecasting, p. 1; p. 5]).

The Park 2016 SHD dataset is almost entirely unexploited for machine
learning despite being open under CC BY and despite carrying 122
citations on Semantic Scholar (queried 2026-05-25). Exactly one
published machine-learning study reuses the dataset itself, in a
high-school research journal and on a same-day random-split
classification task that is not next-day forecasting; the only other
known reuse is a Bachelor's thesis that this benchmark builds on
[spano2026thesis]. The 2025 Cephalalgia narrative review of the
ML-migraine-prediction field [dumkrieger2025review, p. 1] does not cite
Park 2016 anywhere. The benchmark presented here therefore fills a real
gap: a peer-reviewable next-day forecasting evaluation on the largest
open trigger-diary cohort, with the reporting discipline the
prediction-model literature has converged on.

This work develops and evaluates that benchmark in seven layered
contributions. Additions 0 and 1 compare two architecture families
(calibrated XGBoost stacking with and without 500-trial Optuna
hyperparameter search; the TabPFN tabular foundation model across five
released variants) across three feature sets, three split types
(chronological, stratified, patient hold-out), and two split ratios,
with bootstrap CIs and calibration slope reported alongside
discrimination throughout. Addition 2 layers SHAP and ALE explainability
on the headline cells to triangulate the leakage mechanism and
recover Park's univariate trigger ordering. Addition 3 characterises the
temporal dependence of the daily attack series and tests whether a
sequence model is justified; Addition 4 implements that sequence
comparison. Addition 5 reports the per-patient AUROC distribution and
the precision-weighted within-person C-statistic, the metrics
Holsteen et al. argued the field should use. The same addition runs a
leave-one-site-out external check across the two SHD recruitment sites
(Uijeongbu and Dongtan), the strongest external validation achievable
given that no comparable public dataset exists for true out-of-cohort
transport. Addition 6 layers decision-curve net benefit and a Brier
skill score against per-patient climatology, the discipline that
distinguishes forecast *quality* from forecast *value*
[murphy1993forecast, p. 281; vickers2019dca, pp. 1, 3-4]. The whole grid
follows TRIPOD+AI reporting [collins2024tripodAI, p. 6] and PROBAST+AI
risk-of-bias assessment [moons2025probastAI].

The intended use of the benchmark is to characterise the achievable
performance of next-day diary-based migraine and headache forecasting
on the Park 2016 SHD cohort, so that an end-user evaluating whether
diary-only forecasting can support a pre-emptive medication decision
has an honest reference point. Within this study, the intended users
of the prediction models being benchmarked are the patient cohort
itself (for self-directed timing decisions) and any clinician
supervising preventive treatment; deployment-stage user-interaction
considerations are out of scope for a benchmark study and fall under
the DECIDE-AI guideline [vasey2022decideAI]. The applicability domain
is correspondingly narrow: the Park 2016 cohort enrolled from two
Korean university clinics under ICHD-3 episodic-migraine inclusion
criteria, resulting in approximately 82% female enrolment, ages 19-55,
and effectively no within-cohort sociodemographic variation. Sex-based,
ethnicity-based, age-based, or country-of-residence-based health
inequalities cannot be evaluated from this single cohort; the benchmark
therefore makes no fairness claims across underrepresented groups, and
results should be applied only to populations resembling the Park 2016
inclusion criteria.

The paper's load-bearing finding, which the per-Addition layering makes
defensible, is that the pooled AUROC commonly reported in the
diary-forecasting literature overstates within-person forecasting skill
on the Park 2016 cohort. The within-person C-statistic clusters near
0.55, the per-patient AUROC distribution centres near chance, and the
Brier skill against per-patient climatology is negative for migraine.
These findings reproduce the within-cohort Holsteen pattern and
realign the published-AUROC framing for the cohort.
