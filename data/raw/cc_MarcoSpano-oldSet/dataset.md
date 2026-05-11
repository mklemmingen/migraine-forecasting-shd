# Translated and Engineered Dataset - Spano (2026)

## Overview

This document describes the two intermediate dataset files produced by Marco Samuel Spano between September and December 2025, derived from the original SHD source data. These files represent the data pipeline used in Spano's bachelor thesis (Reutlingen University, submitted 30 January 2026) and form the starting point for the benchmark rebuild in this repository.

**Related document:** For the original SHD source file, its three-sheet structure, and the complete transformation pipeline into benchmark training data, see [`SHD-Dataset.md`](SHD-Dataset.md).

---

## Files

| File | Size | Creator | Created | Modified |
|------|------|---------|---------|---------|
| `data/Translated-Dataset.xlsx` | 1,645,395 bytes | Spano, Marco Samuel | 2025-09-22 | 2025-11-03 |
| `data/Engineered-Dataset.xlsx` | 511,815 bytes | - | 2025-11-05 | 2025-12-09 |

---

## Translated-Dataset.xlsx

### What Spano Did

Spano manually translated Sheet 3 ("total diary 4579") of `SHD-Dataset.xls` from Korean into English, producing a flat single-sheet working file. The translation covers column headers and categorical string values. All numeric values and binary flags are carried over unchanged from the source.

A second sheet - Mapings - was added as a patient ID cross-reference table mapping EM-prefixed study identifiers to DHA-prefixed diary identifiers. This sheet has no real header row and must be read with `header=None`. It contains one Korean token (`번호`, meaning "number") which pandas misdetects as a column header.

The Translated file contains 4,454 rows - 137 fewer than the 4,591 raw rows in Sheet 3. The reduction reflects the exclusion of the title row, the two-row header, and aggregate/totals rows that were embedded in the original Excel file.

### Column Structure

All 34 columns are fully populated (4,454/4,454 non-null) except where noted. All trigger columns are binary int64 (0/1).

| Idx | Column | Profile | Notes |
|-----|--------|---------|-------|
| 0 | Entry ID | unique 3,114; min=3 / max=5,140 | |
| 1 | Patient ID | unique 63 raw strings | ⚠ case artifact - see below |
| 2 | Gender | F: 3,629 / M: 825 | |
| 3 | Date | int64 YYYYMMDD; 20140813→20150414 | |
| 4 | Year | 2014: 3,297 / 2015: 1,157 | |
| 5 | Month | 9 months (Aug 2014–Apr 2015) | |
| 6 | Day | 1–31 | |
| 7 | Migraine (Yes/No) | 0: 4,192 / 1: 262 (5.9%) | |
| 8 | Migraine ongoing | 0: 262 / 1: 4,192 | complement of col 7 |
| 9 | Stress | 0: 3,942 / 1: 512 (11.5%) | |
| 10 | Oversleeping | 0: 4,384 / 1: 70 (1.6%) | |
| 11 | Lack of sleep | 0: 4,073 / 1: 381 (8.6%) | |
| 12 | Exercise | 0: 4,326 / 1: 128 (2.9%) | ⚠ semantic shift - see below |
| 13 | No Exercise | 0: 4,287 / 1: 167 (3.7%) | |
| 14 | Physical fatigue | 0: 4,013 / 1: 441 (9.9%) | dropped in Engineered |
| 15 | Menstrual cycle: menstruation | 0: 4,263 / 1: 191 (4.3%) | |
| 16 | Menstrual cycle: ovulation | 0: 4,395 / 1: 59 (1.3%) | |
| 17 | Exessive emotional changes | 0: 4,370 / 1: 84 (1.9%) | ⚠ spelling error; dropped in Engineered |
| 18 | Total (internal triggers) | group sum cols 9–17 | dropped in Engineered |
| 19 | Wheater/temperature change | 0: 4,228 / 1: 226 (5.1%) | ⚠ spelling error - causes weather bug |
| 20 | Excessive sunlight | 0: 4,419 / 1: 35 (0.8%) | dropped in Engineered |
| 21 | Noise | 0: 4,347 / 1: 107 (2.4%) | dropped in Engineered |
| 22 | Inappropriate Lighting | 0: 4,447 / 1: 7 (0.2%) | dropped in Engineered |
| 23 | Specific smells (cosmetic, perfume etc.) | 0: 4,375 / 1: 79 (1.8%) | dropped in Engineered |
| 24 | Total.1 (external triggers) | group sum cols 19–23 | dropped in Engineered |
| 25 | Excessive drinking | 0: 4,404 / 1: 50 (1.1%) | → alcohol_today in Engineered |
| 26 | Irregular Meals (e.g fasting) | 0: 4,260 / 1: 194 (4.4%) | |
| 27 | Overeating | 0: 4,366 / 1: 88 (2.0%) | |
| 28 | Excessive caffeine drinks | 0: 4,249 / 1: 205 (4.6%) | |
| 29 | Excessive smoking | 0: 4,408 / 1: 46 (1.0%) | dropped in Engineered |
| 30 | Cheese, chocolate | 0: 4,425 / 1: 29 (0.7%) | |
| 31 | Travel | 0: 4,424 / 1: 30 (0.7%) | |
| 32 | Other | 0: 4,310 / 1: 144 (3.2%) | dropped in Engineered |
| 33 | Total.2 (other triggers) | group sum cols 25–32 | dropped in Engineered |

### Known Issues

**Col 17 - spelling error:** `Exessive` should be `Excessive`. Carried forward from translation; the value is correct.

**Col 19 - spelling error:** `Wheater/temperature change` should be `Weather/temperature change`. The value is correct (226 positive rows, 5.1% prevalence). This typo is the direct cause of all five weather features being zero in the Engineered file. The feature-engineering code looks up the correctly spelled column name `Weather/temperature change`, fails to find it, and silently fills all weather rows with zero via a zero-guard. Fix: single-character edit to `features/weather.py:3`.

**Patient ID case artifact:** `df["Patient ID"].nunique()` returns 63. `df["Patient ID"].str.upper().nunique()` returns 62. The identifiers `CM-004` and `cm-004` both appear as distinct raw strings; all other patient IDs appear in a single case. This is a data-entry artifact from the source. The cohort contains 62 patients as stated in Park et al. (2016). The Engineered file carries this artifact unchanged.

**Polarity of outcome column:** In `SHD-Dataset.xls`, the column `두통이없는날` means "headache-free day" (Y = no headache). In the Translated file, `Migraine (Yes/No)` uses 1 = migraine present. The polarity was correctly inverted during translation.

**Col 12 - Exercise semantic ambiguity:** In the Original, this column records whether exercise was reported as a migraine trigger on a given day. In the Translated file the header reads simply "Exercise," which is ambiguous between "exercise occurred" and "exercise was a trigger." The Engineered pipeline treats it as a behaviour flag (exercise occurred). This is a semantic shift documented further in the Engineered section below.

---

## Engineered-Dataset.xlsx

### What Spano Did

Spano ran the feature-engineering pipeline (`headfree-backend/model/build_features.py`) on the Translated file, producing the 44-column feature matrix used for model training. The pipeline applied the following transformations:

**Rows:** The last diary entry per patient was dropped because no next-day label exists for that row. This reduced 4,454 rows to 4,391 (−63, one per patient in the 62-patient cohort; the raw patient count of 63 strings produces 63 dropped rows).

**Target construction:** `migraine_target` = `migraine_today.shift(-1)` per patient group. The model predicts whether a migraine will occur on the next day, not the current day. The 29-row difference between `migraine_today` (260 positive) and `migraine_target` (231 positive) represents the last-row drops.

**Feature engineering - 26 new features added across 5 domains:**

| Domain | New features |
|--------|-------------|
| Stress | stress_drop_today, consecutive_stress_days |
| Sleep | any_sleep_issue_today, sleep_debt_3day, sleep_disruption_today, sleep_variability_7day, recent_weekend_sleep_issues |
| Weather | weather_change_today, consecutive_weather_changes, weather_instability_3day, weather_changes_3day_count ⚠ all zero |
| Diet/triggers | consecutive_trigger_days, trigger_foods_today |
| Exercise | consecutive_exercise_days, consecutive_sedentary_days, exercise_days_7day, exercise_consistency_7day, exercise_disruption, travel_exercise_conflict |
| History | migraine_yesterday, migraine_rate_last3, migraine_rate_last7, days_since_last_migraine |
| Context | dow (day of week, 0=Monday) |

**Columns dropped from Translated (13 columns):**

| Column | Translated prevalence | Reason (inferred) |
|--------|----------------------|-------------------|
| Physical fatigue | 9.9% | No replacement created |
| Exessive emotional changes | 1.9% | No replacement created |
| Excessive sunlight | 0.8% | No replacement created |
| Noise | 2.4% | No replacement created |
| Inappropriate Lighting | 0.2% | No replacement created |
| Specific smells | 1.8% | No replacement created |
| Excessive smoking | 1.0% | No replacement created |
| Other | 3.2% | No replacement created |
| Migraine ongoing | complement of migraine_today | Redundant |
| Total (internal) | group sum | Redundant |
| Total.1 (external) | group sum | Redundant |
| Total.2 (other) | group sum | Redundant |
| No Exercise | 3.7% | Replaced by consecutive_sedentary_days |

**Note on exercise semantics:** `exercise_today` in the Engineered file records whether exercise occurred on a given day. This is behaviourally distinct from the original `Exercise` column in the source, which recorded exercise as a migraine trigger. The Engineered column reflects exercise as an activity, not as a precipitant.

### Column Structure

**Identifiers and labels (5):**

| Feature | Type | Profile |
|---------|------|---------|
| Entry ID | int64 | unique 3,081; min=3 / max=5,139 |
| Patient ID | object | unique 63 (case artifact carried) |
| Date | datetime64 | 2014-08-13 → 2015-04-13 |
| migraine_today | int64 | 0: 4,131 / 1: 260 (5.9%) |
| migraine_target | int64 | 0: 4,160 / 1: 231 (5.3%) ← prediction target |

**Stress features (3):**

| Feature | Profile |
|---------|---------|
| Stress / stress_today | 0: 3,888 / 1: 503 (11.5%) - ⚠ duplicate pair |
| stress_drop_today | 0: 4,141 / 1: 250 (5.7%) |
| consecutive_stress_days | min=0 / max=24 / mean=0.3 |

**Sleep features (7):**

| Feature | Profile |
|---------|---------|
| lack_of_sleep_today | 0: 4,017 / 1: 374 (8.5%) |
| oversleeping_today | 0: 4,322 / 1: 69 (1.6%) |
| any_sleep_issue_today | 0: 3,948 / 1: 443 (10.1%) |
| sleep_debt_3day | 0: 3,570 / 1: 629 / 2: 161 / 3: 31 |
| sleep_disruption_today | 0: 4,083 / 1: 308 (7.0%) |
| sleep_variability_7day | float; min=0 / max=0.577 / mean=0.152 |
| recent_weekend_sleep_issues | 0–6; 0: 2,724 |

**Weather features (5) - all zero:**

| Feature | Profile |
|---------|---------|
| Weather/temperature change | ⚠ constant 0 |
| weather_change_today | ⚠ constant 0 |
| consecutive_weather_changes | ⚠ constant 0 |
| weather_instability_3day | ⚠ constant 0 |
| weather_changes_3day_count | ⚠ constant 0 |

Source has 226 positive weather-trigger rows (5.1%). All zeroed due to column-name typo mismatch. See Translated file issue above.

**Dietary and trigger features (8):**

| Feature | Profile |
|---------|---------|
| irregular_meals_today | 0: 4,198 / 1: 193 (4.4%) |
| overeating_today | 0: 4,303 / 1: 88 (2.0%) |
| excessive_caffeine_today | 0: 4,187 / 1: 204 (4.6%) |
| alcohol_today | 0: 4,342 / 1: 49 (1.1%) |
| Cheese, chocolate / trigger_foods_today | 0: 4,362 / 1: 29 (0.7%) - ⚠ duplicate pair |
| Travel / travel_today | 0: 4,361 / 1: 30 (0.7%) - ⚠ duplicate pair |
| consecutive_trigger_days | min=0 / max=19 / mean=0.3 |

**Exercise features (8):**

| Feature | Profile |
|---------|---------|
| exercise_today | 0: 4,266 / 1: 125 (2.8%) |
| consecutive_exercise_days | 0: 4,266 / 1: 88 / 2: 22 / 3: 9 |
| consecutive_sedentary_days | min=0 / max=126 / mean=36.4 |
| exercise_days_7day | 0: 3,993 / 1: 181 / 2: 96 / 3: 66 |
| exercise_consistency_7day | 0: 4,270 / 1: 121 (2.8%) |
| exercise_disruption | 0: 4,274 / 1: 117 (2.7%) |
| travel_exercise_conflict | 0: 4,361 / 1: 30 (0.7%) |

**Hormonal features (2):**

| Feature | Profile |
|---------|---------|
| menstruation_today | 0: 4,203 / 1: 188 (4.3%) |
| ovulation_today | 0: 4,332 / 1: 59 (1.3%) |

**Context and history features (6):**

| Feature | Profile |
|---------|---------|
| dow | 0(Mon)–6(Sun); near-uniform 624–639 per day |
| migraine_yesterday | 0: 4,137 / 1: 254 (5.8%) |
| migraine_rate_last3 | 0.0: 3,837 / 0.333: 398 / 0.667: 106 / 1.0: 46 |
| migraine_rate_last7 | float; mean=0.058 |
| days_since_last_migraine | 1,861 null (42.4%); min=1 / max=60 / mean=15.7 |

### Known Issues

**Weather data absent.** Five weather columns are constant zero. Source has 226 positive rows. Root cause: column-name typo in Translated file (`Wheater` vs `Weather`). Fix applied in benchmark rebuild.

**Three duplicate column pairs.** `Stress`/`stress_today` (identical values), `Travel`/`travel_today` (identical values), `Cheese, chocolate`/`trigger_foods_today` (identical values). All three duplicates are removed in the benchmark rebuild, reducing the effective feature count from 38 to 35.

**days_since_last_migraine: 42% null.** 1,861 null values for patient-days with no prior migraine event in the observation window. Not a data error. Benchmark imputation: filled with 61 (one beyond the observed maximum of 60), flagging "no prior event in window" as a distinct state.

**Four trigger categories absent with no replacement.** Sunlight (0.8%), noise (2.4%), inappropriate lighting (0.2%), and specific smells (1.8%) from the Translated file have no engineered equivalents. Physical fatigue (9.9%) and emotional changes (1.9%) were also dropped without replacement. These features are absent from the benchmark training matrix.

**Reproducibility blocker (at time of audit).** `features/sleep.py:42` and `features/diet.py:76` contained unresolved git merge conflict markers, causing a `SyntaxError` when running `build_features.py`. The existing `engineered.parquet` on disk was generated from an earlier clean state. Both conflicts are resolved in the benchmark rebuild.

---

## Spano Baseline Model

The model trained on the Engineered file and reported in the thesis is a stacked ensemble: XGBoost base learner and L1-regularised logistic regression, combined with blend weight α = 0.10 (selected over linspace(0, 1, 21) by validation MCC). Per-base-learner Platt and isotonic calibrators are fitted on the validation window, followed by a final isotonic recalibrator on blended validation probabilities. Operating thresholds are chosen on the same validation window at MCC-optimal and sensitivity ≥ 0.50.

Split: 85/15 time-forward at cutoff 2015-01-24 (train n = 3,717, val n = 674).

Reported validation metrics (thesis):

| Metric | Value |
|--------|-------|
| AUROC | 0.7505 |
| AUPRC | 0.3159 |
| Brier score | 0.0415 |
| ECE10 (bootstrap mean) | 0.009 |

**Calibration note.** The validation window was used for model selection, calibration fitting, blend weight selection, final recalibrator fitting, and threshold selection. The ECE10 point estimate on the validation window is near machine zero (the isotonic calibrator memorises the window it is fitted on). The bootstrap mean of 0.009 mitigates this by resampling, but the calibration estimate remains optimistic. This is corrected in the benchmark rebuild by separating the calibration and evaluation windows.

---

## Summary: What the Benchmark Changes

| Aspect | Spano (2026)                | This Benchmark                                                                                                                             |
|--------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Weather features | All zero (typo) | Restored (226 positives)                                                                                                                   |
| Duplicate columns | 3 pairs present  | Removed                                                                                                                                    |
| Dropped triggers | 6 categories absent | Absent (not restored)                                                                                                                      |
| days_since_last_migraine nulls | 1,861 null | Filled with 61                                                                                                                             |
| Calibration | Fitted on validation window | Fitted on separate split                                                                                                                   |
| Reproducibility | Merge conflicts in source | Resolved                                                                                                                                   |
| Feature count (effective) | 38 (incl. duplicates) | 35                                                                                                                                         |
| Direct metric comparability | Thesis baseline | ⚠ Not directly comparable - different feature set and evaluation protocol - therefore, see marco blend and feature for individual baseline |

Because the weather correction changes the feature set, no metric from this benchmark is directly numerically comparable to the Spano thesis baseline. The difference is reported explicitly in all experiment result tables.