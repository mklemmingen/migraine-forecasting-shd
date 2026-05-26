# Korean Smartphone Headache Diary - Source Dataset

> Position in the paper: **Data and features (2.1)**. Reads after the
> repository README; precedes `park_features.md`. The engineered-feature
> EPV bands documented here are referenced from every per-Addition methods
> doc.

## Source Publication

**Park J-W, Chu MK, Kim J-M, Park S-G, Cho S-J (2016)**  
Analysis of Trigger Factors in Episodic Migraineurs Using a Smartphone Headache Diary Applications.  
*PLOS ONE* 11(2): e0149577.  
https://doi.org/10.1371/journal.pone.0149577

The SHD dataset is the supplementary file S1 of the above publication, redistributed under the Creative Commons Attribution License (CC BY 4.0) under which the original article was published. Consult the original publication for usage conditions.

### Why this dataset

Park 2016 SHD is the appropriate starting point for this benchmark for four
reasons. First, **it is publicly released under CC BY 4.0** (file S1 of the
PLOS ONE article), so a reproducible benchmark can ship the engineered
features without re-licensing the raw rows. Second, **it has the right
granularity**: daily diary entries for ~85 days per patient give the
patient-day temporal resolution next-day forecasting needs, unlike
millisecond wearable streams (Faisal 2026) or visit-only EHR snapshots.
Third, **it is the right cohort definition**: ICHD-3 episodic migraine,
2-14 headache days per month, formally diagnosed at two Korean university
neurology clinics, with documented inclusion criteria - the population a
clinical pre-emptive-medication forecast would target. Fourth, **the
trigger structure is documented at source**: Park et al. published trigger
odds-ratios in Table 4 of the original paper, giving the benchmark a
ready-made comparator for the SHAP-attribution recovery check (Addition 2
§1b Claim 2). The next-best publicly available cohort with similar
properties (Houle 2005, 132 patients) uses a different diary instrument
and lacks the published trigger-OR table that anchors the Addition 2
Park-rank comparison.

**Related document:** For prior work on the intermediate and engineered representations of this data as produced by Marco Samuel Spano [2], see [`cc_MarcoSpano-oldSet/dataset.md`](data/cc_MarcoSpano-oldSet/dataset.md).

---

## File

`data/raw/SHD-Dataset.xls` - 5,764,608 bytes, last modified 2026-04-29.  
Engine required: xlrd (pin `xlrd < 2.0`).  
Three sheets. All column headers in Korean throughout.

---

## Study Design

Sixty-two episodic migraineurs were recruited between September 2014 and January 2015 at the neurology outpatient clinics of two Korean university hospitals: 의정부성모병원 (32 patients) and 동탄성심병원 (30 patients). Inclusion criteria: age 19–55, ICHD-3 beta-defined migraine with or without aura, 2–14 headache days per month, stable headache characteristics for at least one year prior to enrolment, personal smartphone capable of running the SHD application [1, p. 3].

The cohort was 82.3% female, mean age 37.7 ± 8.6 years, mean illness duration 9.7 ± 8.2 years. Sixty patients had migraine without aura; two had migraine with aura. Baseline clinical instruments: HIT-6 (mean 62.4 ± 9.7), MIDAS (mean 22.0 ± 24.5), HADS-D (mean 9.5), HADS-A (mean 6.5) [1, p. 5, Tab. 1].

Participants logged the SHD application daily for approximately three months. Every day, regardless of headache presence, participants confirmed headache status. On headache days they additionally recorded: pain intensity (VAS 0–10), headache characteristics, associated symptoms, acute medication taken, headache-related functional disability, and the triggers present on the same day. Triggers from the preceding 1–3 days were explicitly excluded [1, p. 4]. Study compliance: 86.3% daily recording rate over 85 ± 13.4 days per patient [1, p. 5].

Eighteen trigger factors were assessed: stress, excessive sleep, sleep deprivation, exercise, fatigue, hormonal changes, emotional changes, weather changes, sunlight, noise, odors, fasting, overeating, caffeine, smoking, alcohol, cheese/chocolate, and traveling [1, p. 3].

Headache classification followed ICHD-3 beta criteria B–D for migraine without aura. Of 1,099 recorded headache days, 336 (30.6%) met migraine criteria and 763 were non-migraine headaches [1, pp. 4–5].

---

# Patient Count Reconciliation

Park et al. reported: 62 patients
Translated (Sheet 3): 63 unique
Diary (engineered):   63 unique
Disability (Sheet 2): 64 unique

In disability but not diary: {'DHA-0045'}
In diary but not disability: set()
In translated but not diary: set()

Patients with only 1 diary entry: 0
Patient IDs: []

---

## Sheet 1 - 62patients

185 columns, 62 patient rows (132 raw rows; ~70 are empty or aggregate rows and should be excluded on read).

**Row structure:** Row 0 is a title row ("S1 File. Dataset of 62 patients"), non-empty in 8/185 cells containing Korean group-header labels. Row 1 is the actual column header row (Korean, 177/185 non-empty). Eight spacer columns at indices 43, 113, 116, 124, 148, 172, 173, 179 are unnamed and used as visual separators.

**Content groups:**

| Group | Cols (approx) | Description |
|-------|---------------|-------------|
| Identifiers | 0–2 | 고유번호 (unique number), 등록번호 (registration ID), 연구번호 (study ID) |
| Demographics | 3–10 | Sex, age, hospital site, height, weight, BMI (+ unknown flags) |
| Study dates | 11–12 | Study start date, diary start date |
| Diagnosis | 13–14 | Migraine type (without/with aura), other diagnosis |
| Headache characteristics | 15–28 | Frequency, illness duration, diary accessibility, duration, severity (categorical + VAS), pulsating, unilateral, aggravation with movement, nausea, vomiting, photophobia, phonophobia, osmophobia |
| Clinical instruments (baseline) | 29–32 | HIT-6, MIDAS, HADS-D, HADS-A |
| Medication | 33–40 | Acute treatment frequency, type; preventive treatment types |
| Trigger endorsements (baseline) | 41–66 | Self-reported trigger list from initial survey; count confirmed in diary; comparison flag; 18 binary trigger flags |
| Attribution | 67 | Trigger attribution percentage |
| Exercise (baseline, IPAQ) | 68–78 | Vigorous/moderate/walking frequency and duration; sedentary time |
| End-of-study | 79–97 | Study end date, duration, weight change, completion status, end-of-study HIT-6/MIDAS, end-of-study IPAQ |
| Compliance and satisfaction | 98–112 | Recording adherence, diary time, non-recording reasons, diary completion time, satisfaction Likert items (7 items) |
| Aggregate diary statistics | 114–184 | Recording rate, total diary days, headache days, headache rate, per-trigger occurrence counts, per-trigger headache co-occurrence counts, exercise totals |

**Selected column profiles:**

| Korean | English | Profile |
|--------|---------|---------|
| 성별 | Sex | 여성(F): 51 / 남성(M): 11 |
| 연령 | Age | min=19 / max=56 / mean=37.3 |
| 병원 | Hospital | 의정부: 32 / 동탄한림: 30 |
| 편두통 진단 | Migraine type | 무조짐(without aura): 60 / 조짐(with aura): 2 |
| 두통빈도(일/월) | Headache freq (days/month) | min=1 / max=30 / mean=6.4 |
| 유병기간(년) | Illness duration (years) | min=1 / max=30 / mean=9.7 |
| 연구시작 - HIT6 | Baseline HIT-6 | mean=62.3 |
| 연구시작 - MIDAS | Baseline MIDAS | mean=22.0 |
| 연구기간 | Study duration (days) | min=35 / max=126 / mean=85.2 |
| 기록 순응도 | Recording adherence | 매일(daily): 36 / ≥절반: 16 / <절반: 3 |

> This sheet is not used in the Stage 0–4 modelling pipeline. It is preserved as clinical reference and is a candidate input for patient-level stratification or personalisation work.

---

## Sheet 2 - headache diary 1099

101 columns, 1,100 rows (1,099 headache events + 1 duplicate entry).

**Row structure:** Row 0 is a title row. Row 1 is a group header (41/101 non-empty). Row 2 is a sub-header (74/101 non-empty). Data begins at Row 3. Multi-row headers are reconstructed as "group | subheader" for programmatic access.

Each row is one headache event, covering both migraine and non-migraine headaches.

**Content groups:**

| Group | Cols | Description |
|-------|------|-------------|
| Identifiers and time | 0–7 | Entry number, registration ID, study ID, start/end datetime (YYYYMMDDHHMI), duration, date (YYYYMMDD) |
| Severity | 8–10 | Categorical (mild/moderate/severe) + binary moderate-or-above + VAS (0–10) |
| Pain location | 11–18 | Left/right/centre/bilateral/eye-area/neck; location count; unilateral flag |
| Headache type | 19–24 | Pulsating/pressing/stabbing/dull; type count; pulsating flag |
| Worsening with movement | 25–26 | Y/N text + binary |
| Associated symptoms | 27–37 | Nausea, phonophobia, vomiting, nausea-or-vomiting, osmophobia, photophobia; counts; photo/phono combined; ICHD major/minor criterion; migraine classification flag |
| Acute medication | 38–49 | Drug slots (6 unnamed columns); effect rating (none/partial/full); medication-taken flags |
| Disability | 50–57 | Overall disability flag; work/school absent; work/school efficiency halved; work/school subtotal; housework unable; housework efficiency halved; housework subtotal; social/leisure missed |
| Trigger factors | 58–84 | Presence flag; count; 18 individual trigger binary flags (internal/external/other groups) |
| Relief factors | 86–92 | Sleep, rest, massage/stretching, exercise, other; count; free-text |
| Exercise | 93–100 | Exercise Y/N; vigorous/moderate intensity flags; duration text; headache-free flag; headache-ongoing flag |

**Disability columns - detail (cols 50–57):**

| Col | Korean | English | Distribution |
|-----|--------|---------|-------------|
| 50 | 장애 | Overall disability | 524/1,099 positive (47.6%) |
| 51 | 학교/직장 - 결근 | Work/school: absent | 22 severe (of 254 work-affected events) |
| 52 | 학교/직장 - 능률 절반 이하 | Work/school: efficiency halved | 232 moderate (of 254 work-affected events) |
| 53 | 학교/직장 합계 | Work/school affected | 254/1,099 (23.1%) |
| 54 | 집안에서 - 전혀 못함 | Housework: unable | 84 severe (of 444 housework-affected events) |
| 55 | 집안에서 - 능률 절반 이하 | Housework: efficiency halved | 360 moderate (of 444 housework-affected events) |
| 56 | 집안에서 합계 | Housework affected | 444/1,099 (40.4%) |
| 57 | 모임/여가 - 참여 못함 | Social/leisure: missed plans | 65/1,099 (5.9%) |

Columns 51/52 are mutually exclusive (three-level ordinal: none/moderate/severe). Columns 54/55 follow the same structure for housework. Column 50 equals 1 wherever column 53, 56, or 57 equals 1. Events link to the daily diary via Patient ID and Date.

**Severity distribution:**

| Severity | Count | Percent |
|----------|-------|---------|
| Mild (약함) | 548 | 49.9% |
| Moderate (중간) | 402 | 36.6% |
| Severe (심함) | 117 | 10.6% |

> This sheet is the data source for Stage 5 (disability prediction feasibility). It links to Sheet 3 via Patient ID and Date. It is excluded from the Stage 0–4 predictor sets.

---

## Sheet 3 - total diary 4579

105 columns, 4,591 rows (4,579 diary days + title row + 2-row header + 1 absorbed totals row).

**Row structure:** Same multi-row structure as Sheet 2. Data begins at Row 3. Includes every diary day - headache and non-headache - for all 62 patients. This is the primary source sheet for the modelling pipeline.

**Key columns not present in Sheet 2:**

| Korean | English | Profile |
|--------|---------|---------|
| 예방약 사용 | Preventive medication taken | 0: 2,726 / 1: 1,853 (40.4%) |
| 지속시간(분) | Headache duration (minutes) | min=0 / max=1,439 / mean=474.6 |
| 타이레놀/복합/트립탄제/기타 | Medication types with dose counts | headache rows only |
| 두통이없는날 | Headache-free day | Y: 3,491 / N: 1,002 |

**Outcome column polarity:** `두통이없는날` means "headache-free day" - Y = no headache. This is directionally opposite to the benchmark target (1 = migraine present) and is inverted during translation.

>  Known data quality issue: trigger factor columns contain a spurious row where each value equals the column-wide count of positives (e.g. `{0:795, 1:304, 304:1}`). A totals row from the original Excel was absorbed as a data row. This row is excluded before analysis by filtering on valid patient IDs.

---

## Benchmark Translation and Engineering Plan

This section describes how this benchmark translates and engineers `SHD-Dataset.xls` into its training-ready feature matrix. This is an independent rebuild - not a patch of the Spano pipeline - and reads directly from the Korean source to avoid carrying forward any translation artifacts.

We name the following steps in-depth, so that future work may peer-review and change approaches when found insufficient.

---

### Step 1 - Translation

Produces `data/translated.parquet` from Sheet 3 of `SHD-Dataset.xls`. Code-driven and fully reproducible. Column headers are read from the Korean source and mapped programmatically.

**Reading procedure:**

1. Open `data/SHD-Dataset.xls` with `xlrd<2.0`, sheet `total diary 4579`.
2. Read with `header=[1, 2]` to capture both header rows; data begins at row 3 (zero-indexed row 2 after the title).
3. Forward-fill the group header (row 1) across columns where Excel merged cells produced empty strings.
4. Combine into single-level headers using the format `{group} - {subheader}` matching the verbatim Korean strings in the column mapping below.
5. Filter rows: keep only rows where `등록번호` matches a valid patient identifier (excludes the absorbed totals row described in Sheet 3 above).
6. Apply the column mapping below; **drop every column not listed in the mapping** (e.g. acute medication slots, raw VAS subcomponents, headache start/end times, free-text fields).
7. Apply the type conversions noted below.

**Type conversions during translation:**

| Source column | Source format | Output type |
|---------------|---------------|-------------|
| 두통이없는날 | Y/N string | int (Y=1, N=0) → `headache_free` |
| 날짜 | int YYYYMMDD | datetime (`pd.to_datetime(s.astype(str), format='%Y%m%d')`) |
| 등록번호 | string with case variants | string, uppercased |
| All trigger flags | int 0/1 | int 0/1 (passthrough) |
| 격렬한운동(분), 중등도운동(분) | int (minutes) or null | int, null-filled with 0 (no exercise = 0 minutes) |

**Patient ID source:** Use Sheet 3 column `등록번호` (registration ID) as `patient_id`. This is the diary-level identifier. Sheet 1's 고유번호 and 연구번호 are patient-level only and not present in Sheet 3.

**Naming convention:** Column names in `translated.parquet` use the base name without suffix (e.g. `stress`, `alcohol`). The engineering step renames current-day trigger features with the `_today` suffix in the engineered sets (e.g. `stress_today`, `alcohol_today`) to distinguish them from derived temporal features. The translation table below shows `translated.parquet` names; the engineering tables show final engineered names.

| Korean header (verbatim) | Benchmark column name | Decision | Reason                                                                                                                                                                   |
|--------------------------|------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 번호 | entry_id | Included | identifier                                                                                                                                                               |
| 고유번호 / 연구번호 | patient_id | Included | unified, uppercased - resolves CM-004/cm-004 artifact                                                                                                                    |
| 날짜 | date | Included | parsed to datetime                                                                                                                                                       |
| 두통이없는날 | headache_free | Included | Y=1; inverted to migraine=1 at engineering step                                                                                                                          |
| 두통지속중여부 | headache_ongoing | Included | structural diary field                                                                                                                                                   |
| 예방약 사용 | preventive_medication | Included | Park et al. demonstrate preventive medication significantly modifies trigger–migraine relationship for stress, overeating, alcohol, and travel [1, Tab. 5, p. 9]           |
| 정도 | severity_category | Included | structural headache characteristic                                                                                                                                       |
| Pain intensity (VAS) | severity_vas | Included | structural headache characteristic                                                                                                                                       |
| 내인적 요인 - 스트레스 | stress | Included | OR 1.8 (95% CI 1.4–2.4, p<0.001); most common trigger on headache days 27.6% [1, pp. 5–7]                                                                 |
| 내인적 요인 - 수면과다 | oversleeping | Included | Part of the 18-trigger inventory; included for symmetry within the sleep-disturbance domain [1, p. 3]                                                    |
| 내인적 요인 - 수면부족 | lack_of_sleep | Included | Common headache trigger 20.4% of days; headache likelihood when present 55.1% [1, pp. 5–6]                                                                |
| 내인적 요인 - 운동 | exercise_as_trigger | Included | associated with migraine at p=0.78; prevalence 1.3–1.5% across both headache types [1, Tab. 4, p. 8]                                                    |
| 내인적 요인 - 운동 안하기 | no_exercise | Included | Part of the 18-trigger inventory; behavioural complement to exercise [1, p. 3]                                                                           |
| 내인적 요인 - 육체적 피로 | physical_fatigue | Included | Common trigger on headache days 20.7%; headache likelihood 48.5%; classified by Park et al. as a modifiable migraine trigger [1, pp. 5–6, 9]           |
| 내인적 요인 - 생리주기 : 월경기 | menstruation | Included | OR 3.5 (95% CI 2.3–5.2, p<0.001); significant regardless of preventive medication [1, pp. 7–9]                                                            |
| 내인적 요인 - 생리주기 : 배란기 | ovulation | Included | Component of the hormonal-changes trigger domain in the SHD instrument [1, p. 3]                                                                         |
| 내인적 요인 - 과도한 감정변화 | emotional_changes | Included | Headache likelihood when present 68.8% - third-highest of all triggers [1, p. 6]                                                                         |
| 외부적 요인 - 날씨/온도 변화 | weather_change | Included | Common trigger 9.9% of headache days; read from Korean header [1, p. 5]                                                                                  |
| 외부적 요인 - 과도한 햇빛 | sunlight | Included | Not significantly associated with migraine (p=0.73); prevalence 0.8%[1, Tab. 4, p. 8]                                                                   |
| 외부적 요인 - 소음 | noise | Included | OR 2.8 (95% CI 1.4–4.9, p=0.002); significant regardless of preventive medication [1, pp. 7–8]                                                            |
| 외부적 요인 - 부적절한 조명 | inappropriate_lighting | Included | Not part of Park et al.'s 18-trigger inventory; prevalence 0.2%  [1, p. 3]                                                                               |
| 외부적 요인 - 특정한 냄새(화장품 향수 등) | specific_smells | Included | Headache likelihood when present 71.8% - second-highest of all triggers; significantly more frequent in migraine (p<0.001) [1, p. 6, Tab. 4]            |
| 기타 - 과도한 음주 | alcohol | Included | OR 2.5 (95% CI 1.3–5.0, p=0.009); highest headache likelihood when present 78.6% [1, pp. 6–7]                                                             |
| 기타 - 불규칙한 식사(공복 등) | irregular_meals | Included | Significantly more frequent in migraine (p=0.003); significant in no-preventive-medication subgroup (p=0.03) [1, Tabs. 4–5, pp. 8–9]                      |
| 기타 - 과식 | overeating | Included | OR 2.4 (95% CI 1.1–5.7, p=0.009) [1, p. 7]                                                                                                               |
| 기타 - 과도한 카페인 음료 | excessive_caffeine | Included | Part of the 18-trigger inventory; not significant in stepwise regression but retained given prior literature support cited by Park et al. [1, p. 3]      |
| 기타 - 과도한 흡연 | excessive_smoking | Included | Not significantly associated with migraine (p=0.73); excluded by Park et al. from subgroup analysis due to insufficient cell counts [1, Tabs. 4–5, pp. 8–9] |
| 기타 - 치즈 초콜릿 | cheese_chocolate | Included | Excluded by Park et al. from subgroup analysis due to insufficient cell counts; prevalence 0.7% [1, Tab. 5, p. 9]                                       |
| 기타 - 여행 | travel | Included | Strongest migraine-associated trigger: OR 6.4 (95% CI 1.2–10.2, p=0.003) [1, p. 7]                                                                       |
| 기타 - 기타 | ~~other_trigger~~ | Excluded | Catch-all free-text category; not part of the 18-trigger inventory and not analysed in Park et al. [1, p. 3]                                             |
| 격렬한 운동(분) | vigorous_exercise_min | Included | Enables exercise as behaviour to be derived separately from exercise as trigger                                                                                          |
| 중등도운동(분) | moderate_exercise_min | Included | Enables exercise as behaviour to be derived separately from exercise as trigger                                                                                          |

### Low-Significance Features Retained

| Feature | Prevalence | Park p-value | Retention rationale |
|---------|-----------|--------------|---------------------|
| exercise_as_trigger_today | 1.3% | 0.78 | Model-driven selection; delayed inflammatory response possible at 1-day lag |
| sunlight_today | 0.8% | 0.73 | Model-driven selection; photophobia is a known migraine feature |
| inappropriate_lighting_today | 0.2% | N/A | Model-driven selection; retained for completeness |
| excessive_smoking_today | - | 0.73 | Model-driven selection |
| cheese_chocolate_today | 0.7% | N/A | Model-driven selection; tyramine hypothesis in migraine literature |

These features are retained despite Park et al.'s same-day significance
tests because: (1) same-day p-values do not measure next-day predictive
power, (2) the models used (XGBoost, TabPFN) handle irrelevant features
through regularization, and (3) SHAP (Addition 2) empirically assesses
their contribution.

**Key translation decisions:**

The weather column is read directly from the Korean header `날씨/온도 변화`. This bypasses the English typo `Wheater/temperature change` introduced in previous work, which caused all five weather features to be zeroed. All 236 weather-trigger rows are retained.

`exercise_as_trigger` is excluded from the final feature set (not significant in Park et al., p=0.78). However, `vigorous_exercise_min` and `moderate_exercise_min` are retained to derive `exercise_today` as a behaviour flag at the engineering step - these are semantically distinct and kept separate.

Six columns are excluded at translation based on Park et al. statistical findings: `sunlight` (p=0.73, 0.8% prevalence), `inappropriate_lighting` (not in 18-trigger inventory, 0.2%), `excessive_smoking` (p=0.73, insufficient cell counts in Park et al. subgroup analysis), `cheese_chocolate` (insufficient cell counts, 0.7%), `exercise_as_trigger` (p=0.78), and `other_trigger` (catch-all, not analysed).

Patient IDs are uppercased on read. This resolves the `CM-004`/`cm-004` case artifact, giving 63 canonical patients throughout the translated diary.

Group-sum columns (합계 columns) are not carried forward; they are derived quantities recomputed during engineering where needed.

The absorbed totals row is excluded by retaining only rows where `patient_id` matches a known patient identifier.

---

### Step 2 - Engineering

Produces separated `train_engineered.parquet`, `val_engineered.parquet`, and `test_engineered.parquet` files from `data/translated.parquet`. All features describe the current diary day; the target describes the next day.

**Operation order (script-level):**

1. Sort by `(patient_id, date)` ascending.
2. Construct `migraine_today` from `headache_free` (polarity inversion).
3. Construct `migraine_target = migraine_today.shift(-1)` per patient group.
4. Drop the last diary entry per patient (rows where `migraine_target` is null after the shift).
5. Construct migraine history features (require `migraine_today`).
6. Construct stress, sleep, weather, dietary, physical-activity, other-trigger temporal features (require sorted per-patient series).
7. Construct interaction features that require both groups (e.g. `sleep_disruption_today` needs `migraine_yesterday` from step 5).
8. Drop structural columns (`headache_ongoing`, `severity_category`, `severity_vas`).
9. Apply chronological 70/15/15 train/val/test split.

**Rolling window edge handling:** All rolling features use `min_periods=1` - partial windows at the start of each patient's series compute over available days. This avoids dropping the first 6 days per patient.

### Gap Awareness

208 of ~4,453 consecutive-day transitions have gaps > 1 day (max 37 days).
Rolling features (`*_last3`, `*_last7`, `*_3day`, `*_7day`) use
`min_periods=1` and compute over whatever data is available, which may be
a single day after a long gap. Two gap-awareness features are provided:

- `days_since_last_record`: number of calendar days since the patient's
  previous diary entry. 1 = continuous; > 1 = gap.
- `recording_gap_flag`: binary indicator (1 if gap > 1 day).

Models should learn to discount rolling features when gap indicators are
high. For LSTM (Addition 4), consider masking or segmenting sequences at
gaps > N days.

**Train/Val/Test Split (Chronological 70/15/15):**

- **Train:** First 70% of unique chronological dates.
- **Validation:** Next 15% of unique dates (used for hyperparameter tuning).
- **Test:** Final 15% of unique dates (held out for final unbiased evaluation).
- Split is applied per-row chronologically, not per-patient. Patients may appear in multiple splits; this is intentional for time-forward clinical evaluation simulating real-world forecasting.

**Target construction:**

`migraine_today` = `NOT headache_free` (polarity corrected).  
`migraine_target` = `migraine_today.shift(-1)` per patient (next-day label).  
Last diary entry per patient dropped - no next-day label available (−63 rows).

**Feature groups:**

*Migraine history (5) - not in Park et al.; required for time-series forecasting:*

| Feature | Description | Justification |
|---------|-------------|---------------|
| migraine_yesterday | migraine_today.shift(+1) per patient | Prior-day state is a standard lagged predictor in clinical time-series |
| migraine_rate_last3 | rolling 3-day mean per patient | Short-window attack rate captures cluster patterns |
| migraine_rate_last7 | rolling 7-day mean per patient | Weekly window captures episodic rhythm |
| days_since_last_migraine | days since last migraine_today=1; null-filled with 61 | Park et al. note one-day diary approach limits causality - lagged state partially addresses this [1, p. 10] |
| headache_free_streak | consecutive headache-free days before current day | Refractory period signal |

*Stress (3) - included; OR 1.8 [1, p. 7]:*

| Feature | Description |
|---------|-------------|
| stress_today | direct |
| stress_drop_today | stress=0 today AND stress=1 yesterday - "let-down" pattern |
| consecutive_stress_days | rolling count of consecutive stress=1 days |

*Sleep (7) - included; headache likelihood 55.1% for sleep deprivation [1, p. 6]:*

| Feature | Description |
|---------|-------------|
| lack_of_sleep_today | direct |
| oversleeping_today | direct |
| any_sleep_issue_today | OR of above two |
| sleep_debt_3day | count of lack_of_sleep in rolling 3-day window |
| sleep_disruption_today | any_sleep_issue AND migraine_yesterday=1 |
| sleep_variability_7day | std of any_sleep_issue over rolling 7-day window |
| recent_weekend_sleep_issues | count of sleep issues on Sat/Sun in last 7 days |

*Weather (5) - included; 9.9% of headache days; fully restored after Spano pipeline bug [1, p. 5]:*

| Feature | Description |
|---------|-------------|
| weather_change_today | direct |
| consecutive_weather_changes | rolling count of weather_change=1 |
| weather_instability_3day | sum of weather_change in rolling 3-day window |
| weather_change_yesterday | weather_change.shift(+1) per patient |
| weather_headache_interaction | weather_change AND migraine_yesterday=1 |

*Dietary and travel triggers (7) - selectively included per Park et al. significance:*

| Feature | Included | Reason |
|---------|----------|--------|
| irregular_meals_today | Y | Significantly more frequent in migraine (p=0.003) [1, Tab. 4] |
| overeating_today | Y | OR 2.4 (p=0.009) [1, p. 7] |
| excessive_caffeine_today | Y | Part of 18-trigger inventory; prior literature support [1, p. 3] |
| alcohol_today | Y | OR 2.5 (p=0.009); highest headache likelihood 78.6% [1, pp. 6–7] |
| travel_today | Y | Strongest migraine-associated trigger: OR 6.4 (95% CI 1.2–10.2, p=0.003) [1, p. 7] |
| cheese_chocolate_today | N! excluded | Insufficient cell counts in Park et al. subgroup analysis; 0.7% prevalence [1, Tab. 5] |
| excessive_smoking_today | N! excluded | Not significant (p=0.73); excluded from Park et al. subgroup analysis [1, Tabs. 4–5] |
| consecutive_trigger_days | Y | Rolling count of any included dietary or travel trigger |

*Physical activity (6) - exercise as trigger excluded; behaviour and sedentary state retained:*

| Feature | Included | Reason |
|---------|----------|--------|
| exercise_today | Y | Derived from vigorous/moderate_exercise_min > 0 - behaviour flag, not trigger flag |
| no_exercise_today | Y | Direct from translation; part of Park et al. 18-trigger inventory as sedentary-day flag [1, p. 3] |
| exercise_as_trigger_today | N! excluded | Not significant (p=0.78) in Park et al. [1, Tab. 4] |
| consecutive_exercise_days | Y | Exercise pattern feature |
| consecutive_sedentary_days | Y | Rolling count of exercise_today=0; temporal extension of no_exercise_today |
| exercise_days_7day | Y | Weekly regularity measure |
| vigorous_exercise_min | Y | Duration in minutes - source of exercise_today |

*Other triggers - selectively included per Park et al. significance:*

| Feature | Included | Reason |
|---------|----------|--------|
| physical_fatigue_today | Y | Common trigger 20.7% of headache days; headache likelihood 48.5% [1, pp. 5–6] |
| emotional_changes_today | Y | Headache likelihood 68.8% - third-highest of all triggers [1, p. 6] |
| noise_today | Y | OR 2.8 (p=0.002); significant regardless of preventive medication [1, pp. 7–8] |
| specific_smells_today | Y | Headache likelihood 71.8% - second-highest; significantly more frequent in migraine (p<0.001) [1, p. 6, Tab. 4] |
| sunlight_today | N! excluded | Not significant (p=0.73); 0.8% prevalence - too sparse for reliable modelling [1, Tab. 4] |
| inappropriate_lighting_today | N! excluded | Not part of Park et al. 18-trigger inventory; 0.2% prevalence [1, p. 3] |

### Prodromal Contamination Risk

Noise, specific smells, and emotional changes are known migraine prodromal
symptoms [3], [4]. Park et al. measured
same-day co-occurrence, where prodrome and headache overlap by definition.
Under the next-day shift, these features describe today's state - if the
patient is already in prodrome, they predict *today's* headache, not
tomorrow's.

These features are retained because:
1. They may still carry next-day signal via multi-day prodrome windows.
2. Empirical importance (SHAP, Addition 2) will reveal whether they
   contribute to next-day prediction or are noise under the shift.
3. Removing them preemptively would discard potentially valid signal.

Interpretation guidance: if SHAP ranks noise/smells/emotional_changes near
zero for next-day prediction but high for same-day, prodromal contamination
is the explanation.

*Hormonal (2) - included; OR 3.5 [1, p. 7]:*

| Feature | Description |
|---------|-------------|
| menstruation_today | direct |
| ovulation_today | direct |

*Preventive medication (1) - included:*

| Feature | Description |
|---------|-------------|
| preventive_medication | Park et al. show this significantly modifies trigger–migraine associations for stress, overeating, alcohol, and travel [1, Tab. 5, p. 9] |

*Context (1):*

| Feature | Description |
|---------|-------------|
| dow | day of week (0=Monday); standard time-series context feature |

**Total effective features: 52**
(includes 2 gap features detailing days since last record)

**Structural columns dropped at engineering:** `headache_ongoing` (redundant with `migraine_today`), `severity_category` and `severity_vas` (populated on headache days only - mostly null on non-headache days, introducing structural missingness correlated with the target).

---

### Step 3 - Stage 5 Supplement (Sheet 2 Disability)

Sheet 2 is processed separately into separated Train, Val, and Test Parquet files and is not included in the Stage 0–4 feature matrix. It is joined to the daily diary by patient ID and date for Stage 5 experiments only.

**Train/Val/Test Split Alignment:**
To ensure zero data leakage and exact temporal alignment with the daily diary features, the disability dataset is split into chronologically identical Train (70%), Validation (15%), and Test (15%) sets using the exact date cutoffs established in Step 2. Orphan patients without daily diary logs are filtered out. Outputs are physically separated into `train_disability.parquet`, `val_disability.parquet`, and `test_disability.parquet`.

**Reading procedure for Sheet 2:**

1. Open `data/SHD-Dataset.xls` with `xlrd<2.0`, sheet `headache diary 1099`.
2. Read with `header=[1, 2]` (data begins at row 3); forward-fill the group header.
3. Combine into single-level headers using the `{group} - {subheader}` convention.
4. Filter rows: keep only rows where `등록번호` matches a valid patient identifier.
5. Apply the column mapping below; drop all unlisted columns.

**Korean → English column mapping for Sheet 2:**

| Korean header (verbatim) | Output column | Notes |
|--------------------------|---------------|-------|
| 번호 | entry_id | identifier; matches Sheet 3 `번호` for join |
| 등록번호 | patient_id | uppercased |
| 날짜 | date | parsed via `format='%Y%m%d'` |
| 정도 | severity_category | mild/moderate/severe |
| Pain intensity (VAS) | severity_vas | 0–10 |
| 동반증상 - 편두통유무 | migraine_flag | ICHD migraine classification |
| 유발요인 - 유발요인 수 | trigger_count | integer trigger count |
| 지속시간(분) | headache_duration_min | duration in minutes |
| 구급약 사용1 - 장애 | disability_any | overall disability flag (col 50) |
| 학교 또는 직장 - 결근하거나 등교하지 못하였다 | disability_work_severe | severe work/school disability (col 51) |
| 학교 또는 직장 - 작업또는 학업능률이 절반 이하로 감소하였다 | disability_work_moderate | moderate work/school disability (col 52) |
| 학교 또는 직장 - 합계 | disability_work_affected | work/school subtotal (col 53) |
| 집안에서 - 가사일을 전혀 할 수 없었다 | disability_housework_severe | severe housework disability (col 54) |
| 집안에서 - 가사의 능률이 절반 이하로 감소하였다 | disability_housework_moderate | moderate housework disability (col 55) |
| 집안에서 - 합계 | disability_housework_affected | housework subtotal (col 56) |
| 모임/여가 활동 - 예정이 있었으나 참여 할 수 없었다 | disability_social | social/leisure missed (col 57) |
| All 18 trigger flag columns (cols 60–84) | trigger flags | retain with same names as in `translated.parquet` for join |

**Columns retained:**

| Feature | Description | Prevalence |
|---------|-------------|-----------|
| disability_any | overall disability flag | 47.6% |
| disability_work_severe | absent from work/school | 2.0% of all events |
| disability_work_moderate | work efficiency halved | 21.1% of all events |
| disability_work_affected | work/school subtotal | 23.1% |
| disability_housework_severe | unable to do housework | 7.6% of all events |
| disability_housework_moderate | housework efficiency halved | 32.7% of all events |
| disability_housework_affected | housework subtotal | 40.4% |
| disability_social | missed social/leisure plans | 5.9% |
| severity_vas | VAS pain intensity (0–10) | mean ≈ 4.3 |
| migraine_flag | ICHD classification (migraine vs non-migraine) | 30.6% |

Additionally retained for join and context: `entry_id`, `patient_id`, `date`, `severity_category`, `trigger_count`, all 18 trigger binary flags, `headache_duration_min`.

---

## Target Definitions

Two independent parquet trees exist under `data/processed/`:

| Directory | Target | Definition | Positive rate | Source |
|-----------|--------|------------|---------------|--------|
| `headache/` | Any headache | `headache_free == 0` | ~23.5% | Sheet 3 |
| `migraine/` | ICHD-3 migraine | `migraine_flag == 1` | ~7-8% | Sheet 2 join |

Both trees use identical column names (`migraine_today`, `migraine_target`,
etc.). The directory disambiguates. Lag features (`migraine_yesterday`,
`migraine_rate_last3`, etc.) are recomputed per target mode - in the
migraine tree, they reflect migraine-only history.

Park et al.'s trigger ORs [1, Tab. 4] are migraine-specific. The `migraine/`
tree is the appropriate match for validating against those findings. The
`headache/` tree provides higher event counts and may yield better-calibrated
models due to more training signal.

### Events-per-variable (EPV) and overfitting risk by feature set

The events-per-variable (EPV) ratio is the standard prediction-model
discipline check: events divided by candidate predictor parameters. The
TRIPOD+AI 2024 reporting guideline [5] explicitly flags EPV-style sample-size
considerations as a required reporting item; the recent Martin et al. 2025
statistical primer [6] surveys the formal sample-size calculation literature
that supersedes the rule of thumb [6, p. 1]. We adopt the conservative
reading consistent with Peduzzi 1996 and Riley 2020: `<10` high
overfitting risk, `10-19` marginal, `>=20` low; modern sample-size
calculations are more nuanced but the bands remain useful as a first-pass
indicator.

On the canonical 70_15_15 chrono split (train = 3941 rows), the four feature
sets we benchmark yield the following EPV against the two targets:

| Feature set | n features | events on migraine (~7.2% pos rate, 287 events) | EPV migraine | events on headache (~24.0% pos rate, 948 events) | EPV headache |
|---|---|---|---|---|---|
| `full_features`         | 52 | 287 | **5.5 (high risk)** | 948 | 18.2 (marginal)   |
| `spano_features`        | 31 | 287 | 9.3 (high risk)     | 948 | 30.6 (low)        |
| `no_rolling_features`   | 26 | 287 | 11.0 (marginal)     | 948 | 36.5 (low)        |
| `park_features`         |  6 | 287 | **47.8 (low)**      | n/a | n/a (migraine-only - see docs/park_features.md) |

Two implications carry into the methodology and discussion:

1. The migraine `full_features` cell is in the **high-risk EPV band**. Any
   model fit on this configuration is at elevated risk of capitalising on
   training noise; reported test-set numbers from this cell should be
   read with the matching reservation. Reporting calibration slope
   alongside discrimination [5, 6] is the discipline check; a slope
   markedly below 1 on test confirms the overfitting fingerprint.
2. The migraine `park_features` cell is **comfortably in the low-risk
   band** at EPV 47.8. This is part of the scientific justification for
   the Park feature set's existence as a benchmark variant - it is the
   only feature set in this study where the migraine target has a
   sample size that the prediction-model methodology literature [6]
   would consider adequate without statistical-shrinkage adjustment.

For the headache target every feature set sits in the low-risk band; the
EPV concern is migraine-specific.

---

### Row Provenance

Upstream counts are stable facts about the source file. Split counts vary by
ratio and strategy - authoritative figures are in `data/processed/dataset_characterization.pdf`
and the per-package `package_report.pdf` files.

| Step | Source | Rows | Delta | Cause |
|------|--------|------|-------|-------|
| Raw source | SHD-Dataset.xls / Sheet 3 | 4,591 | - | Includes title, header, totals rows |
| After cleaning | Sheet 3 data only | 4,579 | −12 | Non-data rows excluded |
| After translation | translated.parquet | 4,579 | 0 | All diary days retained |
| After engineering | diary.parquet | 4,516 | −63 | Last entry per patient dropped (target shift) |
| Disability raw | SHD-Dataset.xls / Sheet 2 | 1,099 | - | Sheet 2 headache events only |
| Split outputs | `data/processed/<ratio>/<scheme>/` | varies | - | See package reports for per-split row counts |

---

### What This Benchmark Uses That Spano Did Not

| Data | Decision | Reason |
|------|----------|--------|
| Weather trigger | Included | Restored after Spano pipeline bug; 9.9% of headache days [1, p. 5] |
| Physical fatigue | Included | Headache likelihood 48.5%; modifiable trigger [1, pp. 5–6] |
| Emotional changes | Included | Headache likelihood 68.8% - third-highest [1, p. 6] |
| Noise | Included | OR 2.8 (p=0.002); significant regardless of preventive medication [1, pp. 7–8] |
| Specific smells | Included | Headache likelihood 71.8%; significant in migraine (p<0.001) [1, Tab. 4] |
| Sunlight | Excluded | Not significant (p=0.73); 0.8% prevalence [1, Tab. 4] |
| Inappropriate lighting | Excluded | Not in Park et al. 18-trigger inventory; 0.2% prevalence |
| Excessive smoking | Excluded | Not significant (p=0.73); insufficient counts for subgroup analysis [1, Tabs. 4–5] |
| Cheese/chocolate | Excluded | Insufficient cell counts; 0.7% prevalence [1, Tab. 5] |
| Exercise as trigger | Excluded | Not significant (p=0.78) [1, Tab. 4] |
| Exercise duration (min) | Included | Enables exercise as behaviour (vigorous/moderate) derived separately from trigger flag |
| Preventive medication | Included | Modifies trigger–migraine relationship significantly [1, Tab. 5] |
| Disability outcomes (Sheet 2) | Included (Stage 5 only) | Sheet not used by Spano; 1,099 labelled events with three-domain disability ratings |

---

## References

Citation keys resolve against [`Sources.bib`](../Sources.bib) at the repository root. Numbering is per-document by order of first appearance, IEEE style.

[1] J.-W. Park, M. K. Chu, J.-M. Kim, S.-G. Park, and S.-J. Cho, "Analysis of trigger factors in episodic migraineurs using a smartphone headache diary applications," *PLOS ONE*, vol. 11, no. 2, p. e0149577, Feb. 2016. doi: [10.1371/journal.pone.0149577](https://doi.org/10.1371/journal.pone.0149577). BibTeX key: `park2016shd`.

[2] M. S. Spano, "Stacked ensemble baselines for next-day migraine forecasting on the SHD cohort," Bachelor's thesis, Reutlingen University, Reutlingen, Germany, 2026. BibTeX key: `spano2026thesis`.

[3] N. J. Giffin, L. Ruggiero, R. B. Lipton, S. D. Silberstein, J. F. Tvedskov, J. Olesen, J. Altman, P. J. Goadsby, and A. Macrae, "Premonitory symptoms in migraine: an electronic diary study," *Neurology*, vol. 60, no. 6, pp. 935–940, Mar. 2003. doi: [10.1212/01.wnl.0000052998.58526.a9](https://doi.org/10.1212/01.wnl.0000052998.58526.a9). BibTeX key: `giffin2003premonitory`.

[4] G. G. Schoonman, D. J. Evers, G. M. Terwindt, J. G. van Dijk, and M. D. Ferrari, "The prevalence of premonitory symptoms in migraine: a questionnaire study in 461 patients," *Cephalalgia*, vol. 26, no. 10, pp. 1209–1213, Oct. 2006. doi: [10.1111/j.1468-2982.2006.01195.x](https://doi.org/10.1111/j.1468-2982.2006.01195.x). BibTeX key: `schoonman2006premonitory`.

[5] G. S. Collins, K. G. M. Moons, P. Dhiman, R. D. Riley, A. L. Beam, B. Van Calster, M. Ghassemi, X. Liu, J. B. Reitsma, M. van Smeden, A.-L. Boulesteix, J. C. Camaradou, L. A. Celi, S. Denaxas, A. K. Denniston, B. Glocker, R. M. Golub, H. Harvey, G. Heinze, M. M. Hoffman, A. P. Kengne, E. Lam, N. Lee, E. W. Loder, L. Maier-Hein, B. A. Mateen, M. M. McCradden, L. Oakden-Rayner, J. Ordish, R. Parnell, S. Rose, K. Singh, L. Wynants, and P. Logullo, "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ*, vol. 385, e078378, Apr. 2024. doi: [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378). BibTeX key: `collins2024tripodAI`.

[6] G. P. Martin, R. D. Riley, J. Ensor, and S. W. Grant, "Statistical primer: sample size considerations for developing and validating clinical prediction models," *European Journal of Cardio-Thoracic Surgery*, vol. 67, no. 5, p. ezaf142, May 2025. doi: [10.1093/ejcts/ezaf142](https://doi.org/10.1093/ejcts/ezaf142). BibTeX key: `martin2025samplesize`.