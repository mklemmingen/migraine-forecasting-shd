# Korean Smartphone Headache Diary : Source Dataset

## Source Publication

**Park J-W, Chu MK, Kim J-M, Park S-G, Cho S-J (2016)**  
Analysis of Trigger Factors in Episodic Migraineurs Using a Smartphone Headache Diary Applications.  
*PLOS ONE* 11(2): e0149577.  
https://doi.org/10.1371/journal.pone.0149577

The SHD dataset is the supplementary file S1 of the above publication, redistributed under the Creative Commons Attribution License (CC BY 4.0) under which the original article was published. Consult the original publication for usage conditions.

**Related document:** For prior work on the intermediate and engineered representations of this data as produced by Marco Samuel Spano (2025–2026), see [`cc_MarcoSpano-oldSet/dataset.md`](data/cc_MarcoSpano-oldSet/dataset.md).

---

## File

`data/SHD-Dataset.xls` — 5,764,608 bytes
Engine required: xlrd (pin `xlrd < 2.0`).  
Three sheets. All column headers in Korean throughout.

---

## Study Design

Sixty-two episodic migraineurs were recruited between September 2014 and January 2015 at the neurology outpatient clinics of two Korean university hospitals: 의정부성모병원 (32 patients) and 동탄성심병원 (30 patients). Inclusion criteria: age 19–55, ICHD-3 beta-defined migraine with or without aura, 2–14 headache days per month, stable headache characteristics for at least one year prior to enrolment, personal smartphone capable of running the SHD application (Park et al., 2016, p. 3).

The cohort was 82.3% female, mean age 37.7 ± 8.6 years, mean illness duration 9.7 ± 8.2 years. Sixty patients had migraine without aura; two had migraine with aura. Baseline clinical instruments: HIT-6 (mean 62.4 ± 9.7), MIDAS (mean 22.0 ± 24.5), HADS-D (mean 9.5), HADS-A (mean 6.5) (Park et al., 2016, p. 5, Table 1).

Participants logged the SHD application daily for approximately three months. Every day, regardless of headache presence, participants confirmed headache status. On headache days they additionally recorded: pain intensity (VAS 0–10), headache characteristics, associated symptoms, acute medication taken, headache-related functional disability, and the triggers present on the same day. Triggers from the preceding 1–3 days were explicitly excluded (Park et al., 2016, p. 4). Study compliance: 86.3% daily recording rate over 85 ± 13.4 days per patient (Park et al., 2016, p. 5).

Eighteen trigger factors were assessed: stress, excessive sleep, sleep deprivation, exercise, fatigue, hormonal changes, emotional changes, weather changes, sunlight, noise, odors, fasting, overeating, caffeine, smoking, alcohol, cheese/chocolate, and traveling (Park et al., 2016, p. 3).

Headache classification followed ICHD-3 beta criteria B–D for migraine without aura. Of 1,099 recorded headache days, 336 (30.6%) met migraine criteria and 763 were non-migraine headaches (Park et al., 2016, p. 4–5).

---

## Sheet 1 — 62patients

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

## Sheet 2 — headache diary 1099

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

**Disability columns — detail (cols 50–57):**

| Col | Korean | English | Distribution |
|-----|--------|---------|-------------|
| 50 | 장애 | Overall disability | 524/1,099 positive (47.6%) |
| 51 | 학교/직장 — 결근 | Work/school: absent | 22 severe (of 254 work-affected events) |
| 52 | 학교/직장 — 능률 절반 이하 | Work/school: efficiency halved | 232 moderate (of 254 work-affected events) |
| 53 | 학교/직장 합계 | Work/school affected | 254/1,099 (23.1%) |
| 54 | 집안에서 — 전혀 못함 | Housework: unable | 84 severe (of 444 housework-affected events) |
| 55 | 집안에서 — 능률 절반 이하 | Housework: efficiency halved | 360 moderate (of 444 housework-affected events) |
| 56 | 집안에서 합계 | Housework affected | 444/1,099 (40.4%) |
| 57 | 모임/여가 — 참여 못함 | Social/leisure: missed plans | 65/1,099 (5.9%) |

Columns 51/52 are mutually exclusive (three-level ordinal: none/moderate/severe). Columns 54/55 follow the same structure for housework. Column 50 equals 1 wherever column 53, 56, or 57 equals 1. Events link to the daily diary via Patient ID and Date.

**Severity distribution:**

| Severity | Count | Percent |
|----------|-------|---------|
| Mild (약함) | 548 | 49.8% |
| Moderate (중간) | 402 | 36.5% |
| Severe (심함) | 117 | 10.6% |

> This sheet was not used in the Spano (2025) thesis pipeline. In this benchmark it is the data source for Stage 5 (disability prediction feasibility). It links to Sheet 3 via Patient ID and Date.

---

## Sheet 3 — total diary 4579

105 columns, 4,591 rows (4,579 diary days + title row + 2-row header + 1 absorbed totals row).

**Row structure:** Same multi-row structure as Sheet 2. Data begins at Row 3. Includes every diary day — headache and non-headache — for all 62 patients. This is the primary source sheet for the modelling pipeline.

**Key columns not present in Sheet 2:**

| Korean | English | Profile |
|--------|---------|---------|
| 예방약 사용 | Preventive medication taken | 0: 2,726 / 1: 1,853 (40.4%) |
| 지속시간(분) | Headache duration (minutes) | min=0 / max=1,439 / mean=474.6 |
| 타이레놀/복합/트립탄제/기타 | Medication types with dose counts | headache rows only |
| 두통이없는날 | Headache-free day | Y: 3,491 / N: 1,002 |

**Outcome column polarity:** `두통이없는날` means "headache-free day" — Y = no headache. This is directionally opposite to the benchmark target (1 = migraine present) and is inverted during translation.

> ⚠ Known data quality issue: trigger factor columns contain a spurious row where each value equals the column-wide count of positives (e.g. `{0:795, 1:304, 304:1}`). A totals row from the original Excel was absorbed as a data row. This row is excluded before analysis by filtering on valid patient IDs.

---

## Benchmark Translation and Engineering Plan

This section describes how this benchmark translates and engineers `SHD-Dataset.xls` into its training-ready feature matrix. This is an independent rebuild — not a patch of the Spano pipeline — and reads directly from the Korean source to avoid carrying forward any translation artifacts.

---

### Step 1 — Translation

Produces `data/translated.parquet` from Sheet 3 of `SHD-Dataset.xls`. Code-driven and fully reproducible. Column headers are read from the Korean source and mapped programmatically.

**Column mapping:**

| Korean header (verbatim) | Benchmark column name | Notes |
|--------------------------|----------------------|-------|
| 번호 | entry_id | |
| 고유번호 / 연구번호 | patient_id | unified, uppercased (resolves CM-004/cm-004 artifact) |
| 날짜 | date | parsed to datetime |
| 두통이없는날 | headache_free | Y=1; inverted to migraine=1 at engineering step |
| 두통지속중여부 | headache_ongoing | |
| 예방약 사용 | preventive_medication | |
| 정도 | severity_category | mild/moderate/severe |
| Pain intensity (VAS) | severity_vas | 0–10 |
| 내인적 요인 — 스트레스 | stress | |
| 내인적 요인 — 수면과다 | oversleeping | |
| 내인적 요인 — 수면부족 | lack_of_sleep | |
| 내인적 요인 — 운동 | exercise_as_trigger | kept as trigger flag, not conflated with behaviour |
| 내인적 요인 — 운동 안하기 | no_exercise | |
| 내인적 요인 — 육체적 피로 | physical_fatigue | retained (dropped by Spano) |
| 내인적 요인 — 생리주기:월경기 | menstruation | |
| 내인적 요인 — 생리주기:배란기 | ovulation | |
| 내인적 요인 — 과도한 감정변화 | emotional_changes | retained (dropped by Spano) |
| 외부적 요인 — 날씨/온도 변화 | weather_change | read from Korean — bypasses English typo entirely |
| 외부적 요인 — 과도한 햇빛 | sunlight | retained (dropped by Spano) |
| 외부적 요인 — 소음 | noise | retained (dropped by Spano) |
| 외부적 요인 — 부적절한 조명 | inappropriate_lighting | retained (dropped by Spano) |
| 외부적 요인 — 특정한 냄새 | specific_smells | retained (dropped by Spano) |
| 기타 — 과도한음주 | alcohol | |
| 기타 — 불규칙한 식사 | irregular_meals | |
| 기타 — 과식 | overeating | |
| 기타 — 과도한 카페인 음료 | excessive_caffeine | |
| 기타 — 과도한 흡연 | excessive_smoking | retained (dropped by Spano) |
| 기타 — 치즈 초콜릿 | cheese_chocolate | |
| 기타 — 여행 | travel | |
| 기타 — 기타 | other_trigger | retained (dropped by Spano) |
| 격렬한운동(분) | vigorous_exercise_min | |
| 중등도운동(분) | moderate_exercise_min | |

**Key translation decisions:**

The weather column is read directly from the Korean header `날씨/온도 변화`. This bypasses the English typo `Wheater/temperature change` introduced in Spano's Translated file, which caused all five weather features to be zeroed. All 226 weather-trigger rows (5.1% prevalence) are retained.

`exercise_as_trigger` and `exercise_today` (derived at engineering step from duration columns) are kept as separate columns. The Spano pipeline conflated these, repurposing the trigger flag as a behaviour flag.

Patient IDs are uppercased on read. This resolves the `CM-004`/`cm-004` case artifact, giving 62 canonical patients throughout.

Group-sum columns (합계 columns) are not carried forward; they are derived quantities recomputed during engineering where needed.

The absorbed totals row is excluded by retaining only rows where `patient_id` matches a known patient identifier.

---

### Step 2 — Engineering

Produces `data/engineered.parquet` from `data/translated.parquet`. All features describe the current diary day; the target describes the next day.

**Target construction:**

`migraine_today` = `NOT headache_free` (polarity corrected).  
`migraine_target` = `migraine_today.shift(-1)` per patient (next-day label).  
Last diary entry per patient dropped — no next-day label available (−62 rows).

**Feature groups:**

*Migraine history (5):*

| Feature | Description |
|---------|-------------|
| migraine_yesterday | migraine_today.shift(+1) per patient |
| migraine_rate_last3 | rolling 3-day mean per patient |
| migraine_rate_last7 | rolling 7-day mean per patient |
| days_since_last_migraine | days since last migraine_today=1 per patient; null-filled with 61 |
| headache_free_streak | consecutive headache-free days before current day |

*Stress (3):*

| Feature | Description |
|---------|-------------|
| stress_today | direct |
| stress_drop_today | stress=0 today AND stress=1 yesterday per patient |
| consecutive_stress_days | rolling count of consecutive stress=1 days |

*Sleep (7):*

| Feature | Description |
|---------|-------------|
| lack_of_sleep_today | direct |
| oversleeping_today | direct |
| any_sleep_issue_today | OR of above two |
| sleep_debt_3day | count of lack_of_sleep in rolling 3-day window |
| sleep_disruption_today | any_sleep_issue AND migraine_yesterday=1 |
| sleep_variability_7day | std of any_sleep_issue over rolling 7-day window |
| recent_weekend_sleep_issues | count of sleep issues on Sat/Sun in last 7 days |

*Weather (5 — fully restored):*

| Feature | Description | Prevalence |
|---------|-------------|-----------|
| weather_change_today | direct (read from Korean header) | 5.1% |
| consecutive_weather_changes | rolling count of weather_change=1 | |
| weather_instability_3day | sum of weather_change in rolling 3-day window | |
| weather_change_yesterday | weather_change.shift(+1) per patient | |
| weather_headache_interaction | weather_change AND migraine_yesterday=1 | |

*Dietary triggers (8):*

| Feature | Description |
|---------|-------------|
| irregular_meals_today | direct |
| overeating_today | direct |
| excessive_caffeine_today | direct |
| alcohol_today | direct |
| cheese_chocolate_today | direct |
| excessive_smoking_today | direct (retained; dropped by Spano) |
| other_trigger_today | direct (retained; dropped by Spano) |
| consecutive_trigger_days | rolling count of any dietary trigger |

*Physical activity (8):*

| Feature | Description |
|---------|-------------|
| exercise_today | vigorous_exercise_min > 0 OR moderate_exercise_min > 0 |
| exercise_as_trigger_today | direct (kept separate from exercise_today) |
| consecutive_exercise_days | rolling count of exercise_today=1 |
| consecutive_sedentary_days | rolling count of exercise_today=0 |
| exercise_days_7day | count of exercise_today=1 in rolling 7-day window |
| exercise_consistency_7day | exercise_days_7day >= 3 |
| exercise_disruption | exercise_today=0 AND exercise_as_trigger_today=1 |
| vigorous_exercise_min | direct (duration in minutes) |

*Retained triggers absent from Spano pipeline (6):*

| Feature | Prevalence |
|---------|-----------|
| physical_fatigue_today | 9.9% |
| emotional_changes_today | 1.9% |
| noise_today | 2.4% |
| sunlight_today | 0.8% |
| specific_smells_today | 1.8% |
| inappropriate_lighting_today | 0.2% |

*Hormonal (2):*

| Feature | Description |
|---------|-------------|
| menstruation_today | direct |
| ovulation_today | direct |

*Context (1):*

| Feature | Description |
|---------|-------------|
| dow | day of week (0=Monday) |

**Total effective features: 44**

---

### Step 3 — Stage 5 Supplement (Sheet 2 Disability)

Sheet 2 is processed separately into `data/disability.parquet` and is not included in the Stage 0–4 feature matrix. It is joined to the daily diary by patient ID and date for Stage 5 experiments only.

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

### Row Provenance

| Step | Source | Rows | Delta | Cause |
|------|--------|------|-------|-------|
| Raw source | SHD-Dataset.xls / Sheet 3 | 4,591 | — | Includes title, header, totals rows |
| After cleaning | Sheet 3 data only | 4,579 | −12 | Non-data rows excluded |
| After translation | translated.parquet | 4,579 | 0 | All diary days retained |
| After engineering | engineered.parquet | ~4,517 | −62 | Last entry per patient dropped |
| Train split | training set | ~3,839 | — | Rows before cutoff 2015-01-24 |
| Validation split | validation set | ~678 | — | Rows from cutoff 2015-01-24 onward |
| Disability set | disability.parquet | 1,099 | — | Sheet 2 headache events only |

> Exact post-engineering row counts depend on per-patient edge handling and will be updated once the pipeline is run.

---

### What This Benchmark Uses That Spano Did Not

| Data | Source | Why it was unused before |
|------|--------|--------------------------|
| Weather trigger | Sheet 3, Korean col | English column name typo in lookup |
| Physical fatigue | Sheet 3 | Dropped, no replacement |
| Emotional changes | Sheet 3 | Dropped, no replacement |
| Sunlight, noise, lighting, smells | Sheet 3 | All dropped, no replacements |
| Excessive smoking | Sheet 3 | Dropped, no replacement |
| Exercise duration (min) | Sheet 3 | Not extracted |
| Disability outcomes | Sheet 2 | Sheet not used at all |
| Exercise-as-trigger (separate from behaviour) | Sheet 3 | Semantics conflated |

---

## Citation

Park, J.-W., Chu, M. K., Kim, J.-M., Park, S.-G., & Cho, S.-J. (2016). Analysis of trigger factors in episodic migraineurs using a smartphone headache diary applications. *PLOS ONE*, 11(2), e0149577. https://doi.org/10.1371/journal.pone.0149577