"""
Sheet 3 — total diary 4579

105 columns, 4,591 rows (4,579 diary days + title row + 2-row header + 1 absorbed totals row).

Row structure: Same multi-row structure as Sheet 2. Data begins at Row 
3. Includes every diary day — headache and non-headache — for all 62 patients. 
This is the primary source sheet for the modelling pipeline.

Key columns not present in Sheet 2:

| Korean | English | Profile |
|--------|---------|---------|
| 예방약 사용 | Preventive medication taken | 0: 2,726 / 1: 1,853 (40.4%) |
| 지속시간(분) | Headache duration (minutes) | min=0 / max=1,439 / mean=474.6 |
| 타이레놀/복합/트립탄제/기타 | Medication types with dose counts | headache rows only |
| 두통이없는날 | Headache-free day | Y: 3,491 / N: 1,002 |

Outcome column polarity: `두통이없는날` means "headache-free day" — Y = no headache. 
This is directionally opposite to the benchmark target (1 = migraine present) and is inverted during translation.

>  Known data quality issue: trigger factor columns contain a spurious row where each value 
equals the column-wide count of positives (e.g. `{0:795, 1:304, 304:1}`). 
A totals row from the original Excel was absorbed as a data row. 
This row is excluded before analysis by filtering on valid patient IDs.
"""

# package imports
import pandas as pd
import numpy as np
from datetime import timedelta

# Timesplit Variables
global DATE_70, DATE_85

def _flatten_multiindex_columns(df):
    """Helper to forward-fill and flatten MultiIndex columns into '{group} — {subheader}' format."""
    if isinstance(df.columns, pd.MultiIndex):
        # Forward fill level 0 (Group headers)
        level_0 = pd.Series(df.columns.get_level_values(0)).replace(r'^Unnamed:.*', np.nan, regex=True).ffill()
        level_1 = df.columns.get_level_values(1)

        new_cols = []
        for g, s in zip(level_0, level_1):
            if pd.isna(g) or g == s or str(s).startswith('Unnamed:'):
                new_cols.append(str(g) if pd.notna(g) else str(s))
            else:
                new_cols.append(f"{g} — {s}")
        df.columns = new_cols
    return df

"""
Step 1 — Translation

Produces `data/translated.parquet` from Sheet 3 of `SHD-Dataset.xls`. Code-driven and fully reproducible. Column headers are read from the Korean source and mapped programmatically.

Reading procedure:

1. Open `data/SHD-Dataset.xls` with `xlrd<2.0`, sheet `total diary 4579`.
2. Read with `header=[1, 2]` to capture both header rows; data begins at row 3 (zero-indexed row 2 after the title).
3. Forward-fill the group header (row 1) across columns where Excel merged cells produced empty strings.
4. Combine into single-level headers using the format `{group} — {subheader}` matching the verbatim Korean strings in the column mapping below.
5. Filter rows: keep only rows where `등록번호` matches a valid patient identifier (excludes the absorbed totals row described in Sheet 3 above).
6. Apply the column mapping below; drop every column not listed in the mapping (e.g. acute medication slots, raw VAS subcomponents, headache start/end times, free-text fields).
7. Apply the type conversions noted below.

Type conversions during translation:

| Source column | Source format | Output type |
|---------------|---------------|-------------|
| 두통이없는날 | Y/N string | int (Y=1, N=0) → `headache_free` |
| 날짜 | int YYYYMMDD | datetime (`pd.to_datetime(s.astype(str), format='%Y%m%d')`) |
| 등록번호 | string with case variants | string, uppercased |
| All trigger flags | int 0/1 | int 0/1 (passthrough) |
| 격렬한운동(분), 중등도운동(분) | int (minutes) or null | int, null-filled with 0 (no exercise = 0 minutes) |

Patient ID source: Use Sheet 3 column `등록번호` (registration ID) as `patient_id`. This is the diary-level identifier. Sheet 1's 고유번호 and 연구번호 are patient-level only and not present in Sheet 3.

Naming convention: Column names in `translated.parquet` use the base name without suffix (e.g. `stress`, `alcohol`). The engineering step renames current-day trigger features with the `_today` suffix in `engineered.parquet` (e.g. `stress_today`, `alcohol_today`) to distinguish them from derived temporal features. The translation table below shows `translated.parquet` names; the engineering tables show final `engineered.parquet` names.

| Korean header (verbatim) | Benchmark column name | Decision | Reason |
|--------------------------|----------------------|----------|--------|
| 번호 | entry_id | Included | identifier |
| 고유번호 / 연구번호 | patient_id | Included | unified, uppercased — resolves CM-004/cm-004 artifact |
| 날짜 | date | Included | parsed to datetime |
| 두통이없는날 | headache_free | Included | Y=1; inverted to migraine=1 at engineering step |
| 두통지속중여부 | headache_ongoing | Included | structural diary field |
| 예방약 사용 | preventive_medication | Included | Park et al. demonstrate preventive medication significantly modifies trigger–migraine relationship for stress, overeating, alcohol, and travel (Table 5, p. 9) |
| 정도 | severity_category | Included | structural headache characteristic |
| Pain intensity (VAS) | severity_vas | Included | structural headache characteristic |
| 내인적 요인 — 스트레스 | stress | Included | OR 1.8 (95% CI 1.4–2.4, p<0.001); most common trigger on headache days 27.6% (Park et al., 2016, p. 5–7) |
| 내인적 요인 — 수면과다 | oversleeping | Included | Part of the 18-trigger inventory; included for symmetry within the sleep-disturbance domain (Park et al., 2016, p. 3) |
| 내인적 요인 — 수면부족 | lack_of_sleep | Included | Common headache trigger 20.4% of days; headache likelihood when present 55.1% (Park et al., 2016, p. 5–6) |
| 내인적 요인 — 운동 | exercise_as_trigger | Excluded | Not significantly associated with migraine (p=0.78); prevalence 1.3–1.5% across both headache types (Park et al., 2016, Table 4, p. 8) |
| 내인적 요인 — 운동 안하기 | no_exercise | Included | Part of the 18-trigger inventory; behavioural complement to exercise (Park et al., 2016, p. 3) |
| 내인적 요인 — 육체적 피로 | physical_fatigue | Included | Common trigger on headache days 20.7%; headache likelihood 48.5%; classified by Park et al. as a modifiable migraine trigger (Park et al., 2016, p. 5–6, p. 9) |
| 내인적 요인 — 생리주기:월경기 | menstruation | Included | OR 3.5 (95% CI 2.3–5.2, p<0.001); significant regardless of preventive medication (Park et al., 2016, p. 7–9) |
| 내인적 요인 — 생리주기:배란기 | ovulation | Included | Component of the hormonal-changes trigger domain in the SHD instrument (Park et al., 2016, p. 3) |
| 내인적 요인 — 과도한 감정변화 | emotional_changes | Included | Headache likelihood when present 68.8% — third-highest of all triggers (Park et al., 2016, p. 6) |
| 외부적 요인 — 날씨/온도 변화 | weather_change | Included | Common trigger 9.9% of headache days; read from Korean header — bypasses English typo that zeroed this feature in Spano's pipeline (Park et al., 2016, p. 5) |
| 외부적 요인 — 과도한 햇빛 | ~~sunlight~~ | Excluded | Not significantly associated with migraine (p=0.73); prevalence 0.8% — insufficient events for reliable modelling at this cohort size (Park et al., 2016, Table 4, p. 8) |
| 외부적 요인 — 소음 | noise | Included | OR 2.8 (95% CI 1.4–4.9, p=0.002); significant regardless of preventive medication (Park et al., 2016, p. 7–8) |
| 외부적 요인 — 부적절한 조명 | ~~inappropriate_lighting~~ | Excluded | Not part of Park et al.'s 18-trigger inventory; prevalence 0.2% — below the threshold for reliable modelling (Park et al., 2016, p. 3) |
| 외부적 요인 — 특정한 냄새 | specific_smells | Included | Headache likelihood when present 71.8% — second-highest of all triggers; significantly more frequent in migraine (p<0.001) (Park et al., 2016, p. 6, Table 4) |
| 기타 — 과도한음주 | alcohol | Included | OR 2.5 (95% CI 1.3–5.0, p=0.009); highest headache likelihood when present 78.6% (Park et al., 2016, p. 6–7) |
| 기타 — 불규칙한 식사 | irregular_meals | Included | Significantly more frequent in migraine (p=0.003); significant in no-preventive-medication subgroup (p=0.03) (Park et al., 2016, Table 4–5, p. 8–9) |
| 기타 — 과식 | overeating | Included | OR 2.4 (95% CI 1.1–5.7, p=0.009) (Park et al., 2016, p. 7) |
| 기타 — 과도한 카페인 음료 | excessive_caffeine | Included | Part of the 18-trigger inventory; not significant in stepwise regression but retained given prior literature support cited by Park et al. (Park et al., 2016, p. 3) |
| 기타 — 과도한 흡연 | ~~excessive_smoking~~ | Excluded | Not significantly associated with migraine (p=0.73); excluded by Park et al. from subgroup analysis due to insufficient cell counts (Park et al., 2016, Table 4–5, p. 8–9) |
| 기타 — 치즈 초콜릿 | ~~cheese_chocolate~~ | Excluded | Excluded by Park et al. from subgroup analysis due to insufficient cell counts; prevalence 0.7% — too sparse for reliable modelling (Park et al., 2016, Table 5, p. 9) |
| 기타 — 여행 | travel | Included | Strongest migraine-associated trigger: OR 6.4 (95% CI 1.2–10.2, p=0.003) (Park et al., 2016, p. 7) |
| 기타 — 기타 | ~~other_trigger~~ | Excluded | Catch-all free-text category; not part of the 18-trigger inventory and not analysed in Park et al. (Park et al., 2016, p. 3) |
| 격렬한운동(분) | vigorous_exercise_min | Included | Enables exercise as behaviour to be derived separately from exercise as trigger |
| 중등도운동(분) | moderate_exercise_min | Included | Enables exercise as behaviour to be derived separately from exercise as trigger |

Key translation decisions:

The weather column is read directly from the Korean header `날씨/온도 변화`. This bypasses the English typo `Wheater/temperature change` introduced in Spano's Translated file, which caused all five weather features to be zeroed. All 226 weather-trigger rows (5.1% prevalence) are retained.

`exercise_as_trigger` is excluded from the final feature set (not significant in Park et al., p=0.78). However, `vigorous_exercise_min` and `moderate_exercise_min` are retained to derive `exercise_today` as a behaviour flag at the engineering step — these are semantically distinct and kept separate.

Six columns are excluded at translation based on Park et al. statistical findings: `sunlight` (p=0.73, 0.8% prevalence), `inappropriate_lighting` (not in 18-trigger inventory, 0.2%), `excessive_smoking` (p=0.73, insufficient cell counts in Park et al. subgroup analysis), `cheese_chocolate` (insufficient cell counts, 0.7%), `exercise_as_trigger` (p=0.78), and `other_trigger` (catch-all, not analysed).

Patient IDs are uppercased on read. This resolves the `CM-004`/`cm-004` case artifact, giving 62 canonical patients throughout.

Group-sum columns (합계 columns) are not carried forward; they are derived quantities recomputed during engineering where needed.

The absorbed totals row is excluded by retaining only rows where `patient_id` matches a known patient identifier.
"""
def translate_sheet3(raw_df):
    """Step 1 — Translation"""
    df = raw_df.copy()
    df = _flatten_multiindex_columns(df)

    # Verbatim Korean -> Benchmark English mapping
    col_map = {
        '번호': 'entry_id',
        '등록번호': 'patient_id',
        '날짜': 'date',
        '두통이없는날': 'headache_free',
        '두통지속중여부': 'headache_ongoing',
        '예방약 사용': 'preventive_medication',
        '정도': 'severity_category',
        'Pain intensity (VAS)': 'severity_vas',
        '내인적 요인 — 스트레스': 'stress',
        '내인적 요인 — 수면과다': 'oversleeping',
        '내인적 요인 — 수면부족': 'lack_of_sleep',
        '내인적 요인 — 운동 안하기': 'no_exercise',
        '내인적 요인 — 육체적 피로': 'physical_fatigue',
        '내인적 요인 — 생리주기 : 월경기': 'menstruation',
        '내인적 요인 — 생리주기 : 배란기': 'ovulation',
        '내인적 요인 — 과도한 감정변화': 'emotional_changes',
        '외부적 요인 — 날씨/온도 변화': 'weather_change',
        '외부적 요인 — 소음': 'noise',
        '외부적 요인 — 특정한 냄새(화장품 향수 등)': 'specific_smells',
        '기타 — 과도한 음주': 'alcohol',
        '기타 — 불규칙한 식사(공복 등)': 'irregular_meals',
        '기타 — 과식': 'overeating',
        '기타 — 과도한 카페인 음료': 'excessive_caffeine',
        '기타 — 여행': 'travel',
        '격렬한 운동(분)': 'vigorous_exercise_min',
        '중등도운동(분)': 'moderate_exercise_min'
    }

    # Filter and rename columns
    keep_cols = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(keep_cols.keys())].rename(columns=keep_cols)

    # Remove absorbed totals row & NaNs in patient_id
    df = df.dropna(subset=['patient_id'])
    df = df[~df['patient_id'].astype(str).str.contains('합계|total', case=False, na=False)]

    # Standardize types and formats
    df['patient_id'] = df['patient_id'].astype(str).str.upper().str.strip()
    df['date'] = pd.to_datetime(df['date'].astype(str).str.extract(r'(\d{8})')[0], format='%Y%m%d')
    df['headache_free'] = df['headache_free'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

    # Null-fill exercise minutes
    for col in ['vigorous_exercise_min', 'moderate_exercise_min']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Ensure all trigger flags are integers
    trigger_cols = [v for k, v in col_map.items() if v not in (
        'entry_id', 'patient_id', 'date', 'headache_free', 'headache_ongoing',
        'preventive_medication', 'severity_category', 'severity_vas',
        'vigorous_exercise_min', 'moderate_exercise_min'
    )]

    for col in trigger_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df

"""
Step 2 — Engineering

Produces separated `train_engineered.parquet`, `val_engineered.parquet`, and `test_engineered.parquet` files from `data/translated.parquet`. All features describe the current diary day; the target describes the next day.

Operation order (script-level):

1. Sort by `(patient_id, date)` ascending.
2. Construct `migraine_today` from `headache_free` (polarity inversion).
3. Construct `migraine_target = migraine_today.shift(-1)` per patient group.
4. Drop the last diary entry per patient (rows where `migraine_target` is null after the shift).
5. Construct migraine history features (require `migraine_today`).
6. Construct stress, sleep, weather, dietary, physical-activity, other-trigger temporal features (require sorted per-patient series).
7. Construct interaction features that require both groups (e.g. `sleep_disruption_today` needs `migraine_yesterday` from step 5).
8. Drop structural columns (`headache_ongoing`, `severity_category`, `severity_vas`).
9. Apply chronological 70/15/15 train/val/test split.

Rolling window edge handling: All rolling features use `min_periods=1` — partial windows at the start of each patient's 
series compute over available days. This avoids dropping the first 6 days per patient.

Train/Val/Test Split (Chronological 70/15/15):

- Train: First 70% of unique chronological dates.
- Validation: Next 15% of unique dates (used for hyperparameter tuning).
- Test: Final 15% of unique dates (held out for final unbiased evaluation).
- Split is applied per-row chronologically, not per-patient. Patients may appear in multiple splits; this is intentional for time-forward clinical evaluation simulating real-world forecasting.

Target construction:

`migraine_today` = `NOT headache_free` (polarity corrected).  
`migraine_target` = `migraine_today.shift(-1)` per patient (next-day label).  
Last diary entry per patient dropped — no next-day label available (−62 rows).

Feature groups:

*Migraine history (5) — not in Park et al.; required for time-series forecasting:*

| Feature | Description | Justification |
|---------|-------------|---------------|
| migraine_yesterday | migraine_today.shift(+1) per patient | Prior-day state is a standard lagged predictor in clinical time-series |
| migraine_rate_last3 | rolling 3-day mean per patient | Short-window attack rate captures cluster patterns |
| migraine_rate_last7 | rolling 7-day mean per patient | Weekly window captures episodic rhythm |
| days_since_last_migraine | days since last migraine_today=1; null-filled with 61 | Park et al. note one-day diary approach limits causality — lagged state partially addresses this (Park et al., 2016, p. 10) |
| headache_free_streak | consecutive headache-free days before current day | Refractory period signal |

*Stress (3) — included; OR 1.8 (Park et al., 2016, p. 7):*

| Feature | Description |
|---------|-------------|
| stress_today | direct |
| stress_drop_today | stress=0 today AND stress=1 yesterday — "let-down" pattern |
| consecutive_stress_days | rolling count of consecutive stress=1 days |

*Sleep (7) — included; headache likelihood 55.1% for sleep deprivation (Park et al., 2016, p. 6):*

| Feature | Description |
|---------|-------------|
| lack_of_sleep_today | direct |
| oversleeping_today | direct |
| any_sleep_issue_today | OR of above two |
| sleep_debt_3day | count of lack_of_sleep in rolling 3-day window |
| sleep_disruption_today | any_sleep_issue AND migraine_yesterday=1 |
| sleep_variability_7day | std of any_sleep_issue over rolling 7-day window |
| recent_weekend_sleep_issues | count of sleep issues on Sat/Sun in last 7 days |

*Weather (5) — included; 9.9% of headache days; fully restored after Spano pipeline bug (Park et al., 2016, p. 5):*

| Feature | Description |
|---------|-------------|
| weather_change_today | direct (read from Korean header — 226 positives, 5.1%) |
| consecutive_weather_changes | rolling count of weather_change=1 |
| weather_instability_3day | sum of weather_change in rolling 3-day window |
| weather_change_yesterday | weather_change.shift(+1) per patient |
| weather_headache_interaction | weather_change AND migraine_yesterday=1 |

*Dietary and travel triggers (7) — selectively included per Park et al. significance:*

| Feature | Included | Reason |
|---------|----------|--------|
| irregular_meals_today | Y | Significantly more frequent in migraine (p=0.003) (Park et al., 2016, Table 4) |
| overeating_today | Y | OR 2.4 (p=0.009) (Park et al., 2016, p. 7) |
| excessive_caffeine_today | Y | Part of 18-trigger inventory; prior literature support (Park et al., 2016, p. 3) |
| alcohol_today | Y | OR 2.5 (p=0.009); highest headache likelihood 78.6% (Park et al., 2016, p. 6–7) |
| travel_today | Y | Strongest migraine-associated trigger: OR 6.4 (95% CI 1.2–10.2, p=0.003) (Park et al., 2016, p. 7) |
| cheese_chocolate_today | N! excluded | Insufficient cell counts in Park et al. subgroup analysis; 0.7% prevalence (Park et al., 2016, Table 5) |
| excessive_smoking_today | N! excluded | Not significant (p=0.73); excluded from Park et al. subgroup analysis (Park et al., 2016, Table 4–5) |
| consecutive_trigger_days | Y | Rolling count of any included dietary or travel trigger |

*Physical activity (6) — exercise as trigger excluded; behaviour and sedentary state retained:*

| Feature | Included | Reason |
|---------|----------|--------|
| exercise_today | Y | Derived from vigorous/moderate_exercise_min > 0 — behaviour flag, not trigger flag |
| no_exercise_today | Y | Direct from translation; part of Park et al. 18-trigger inventory as sedentary-day flag (Park et al., 2016, p. 3) |
| exercise_as_trigger_today | N! excluded | Not significant (p=0.78) in Park et al. (Park et al., 2016, Table 4) |
| consecutive_exercise_days | Y | Exercise pattern feature |
| consecutive_sedentary_days | Y | Rolling count of exercise_today=0; temporal extension of no_exercise_today |
| exercise_days_7day | Y | Weekly regularity measure |
| vigorous_exercise_min | Y | Duration in minutes — source of exercise_today |

*Other triggers — selectively included per Park et al. significance:*

| Feature | Included | Reason |
|---------|----------|--------|
| physical_fatigue_today | Y | Common trigger 20.7% of headache days; headache likelihood 48.5% (Park et al., 2016, p. 5–6) |
| emotional_changes_today | Y | Headache likelihood 68.8% — third-highest of all triggers (Park et al., 2016, p. 6) |
| noise_today | Y | OR 2.8 (p=0.002); significant regardless of preventive medication (Park et al., 2016, p. 7–8) |
| specific_smells_today | Y | Headache likelihood 71.8% — second-highest; significantly more frequent in migraine (p<0.001) (Park et al., 2016, p. 6, Table 4) |
| sunlight_today | N! excluded | Not significant (p=0.73); 0.8% prevalence — too sparse for reliable modelling (Park et al., 2016, Table 4) |
| inappropriate_lighting_today | N! excluded | Not part of Park et al. 18-trigger inventory; 0.2% prevalence (Park et al., 2016, p. 3) |

*Hormonal (2) — included; OR 3.5 (Park et al., 2016, p. 7):*

| Feature | Description |
|---------|-------------|
| menstruation_today | direct |
| ovulation_today | direct |

*Preventive medication (1) — included:*

| Feature | Description |
|---------|-------------|
| preventive_medication | Park et al. show this significantly modifies trigger–migraine associations for stress, overeating, alcohol, and travel (Park et al., 2016, Table 5, p. 9) |

*Context (1):*

| Feature | Description |
|---------|-------------|
| dow | day of week (0=Monday); standard time-series context feature |

Total effective features: 40

Excluded at engineering (summary): exercise_as_trigger (p=0.78), cheese_chocolate (0.7%, insufficient counts), 
excessive_smoking (p=0.73), sunlight (p=0.73, 0.8%), inappropriate_lighting (not in inventory, 0.2%), 
other_trigger (unstructured catch-all).

Structural columns dropped at engineering: `headache_ongoing` (redundant with `migraine_today`), 
`severity_category` and `severity_vas` (populated on headache days only — mostly null on non-headache days, 
introducing structural missingness correlated with the target).
"""
def engineer_features(translated_df):
    """Step 2 — Engineering"""
    # 1. Sort by patient_id and date
    df = translated_df.sort_values(['patient_id', 'date']).copy()

    # Helper for consecutive counts
    def count_consecutive(s, val):
        is_val = (s == val).astype(int)
        blocks = (is_val == 0).cumsum()
        return is_val.groupby(blocks).cumsum()

    # 2. Construct migraine_today
    df['migraine_today'] = (df['headache_free'] == 0).astype(int)

    # 3. Construct migraine_target
    df['migraine_target'] = df.groupby('patient_id')['migraine_today'].shift(-1)

    # 4. Drop the last diary entry per patient
    df = df.dropna(subset=['migraine_target']).copy()

    # Re-establish groupby for temporal features
    grp = df.groupby('patient_id')

    # 5. Migraine history features
    df['migraine_yesterday'] = grp['migraine_today'].shift(1).fillna(0).astype(int)
    df['migraine_rate_last3'] = grp['migraine_today'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df['migraine_rate_last7'] = grp['migraine_today'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)

    df['headache_free_streak'] = grp['migraine_today'].apply(lambda x: count_consecutive(x, 0)).reset_index(0,
                                                                                                            drop=True)

    last_migraine = df['date'].where(df['migraine_today'] == 1)
    last_migraine_ffill = last_migraine.groupby(df['patient_id']).ffill()
    df['days_since_last_migraine'] = (df['date'] - last_migraine_ffill).dt.days.fillna(61).astype(int)

    # 6 & 7. Construct trigger and interaction features
    # --- Stress ---
    df['stress_today'] = df['stress']
    stress_yesterday = grp['stress_today'].shift(1).fillna(0).astype(int)
    df['stress_drop_today'] = ((df['stress_today'] == 0) & (stress_yesterday == 1)).astype(int)
    df['consecutive_stress_days'] = grp['stress_today'].apply(lambda x: count_consecutive(x, 1)).reset_index(0,
                                                                                                             drop=True)

    # --- Sleep ---
    df['lack_of_sleep_today'] = df['lack_of_sleep']
    df['oversleeping_today'] = df['oversleeping']
    df['any_sleep_issue_today'] = (df['lack_of_sleep_today'] | df['oversleeping_today']).astype(int)

    df['sleep_debt_3day'] = grp['lack_of_sleep_today'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
    df['sleep_disruption_today'] = (df['any_sleep_issue_today'] & df['migraine_yesterday']).astype(int)
    df['sleep_variability_7day'] = grp['any_sleep_issue_today'].rolling(7, min_periods=1).std().reset_index(0,
                                                                                                            drop=True).fillna(
        0)

    is_weekend = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['recent_weekend_sleep_issues'] = (df['any_sleep_issue_today'] & is_weekend).groupby(df['patient_id']).rolling(7,
                                                                                                                     min_periods=1).sum().reset_index(
        0, drop=True)

    # --- Weather ---
    df['weather_change_today'] = df['weather_change']
    df['consecutive_weather_changes'] = grp['weather_change_today'].apply(
        lambda x: count_consecutive(x, 1)).reset_index(0, drop=True)
    df['weather_instability_3day'] = grp['weather_change_today'].rolling(3, min_periods=1).sum().reset_index(0,
                                                                                                             drop=True)
    df['weather_change_yesterday'] = grp['weather_change_today'].shift(1).fillna(0).astype(int)
    df['weather_headache_interaction'] = (df['weather_change_today'] & df['migraine_yesterday']).astype(int)

    # --- Dietary & Travel ---
    diet_travel_cols = ['irregular_meals', 'overeating', 'excessive_caffeine', 'alcohol', 'travel']
    for col in diet_travel_cols:
        df[f'{col}_today'] = df[col]

    any_diet_travel = df[[f'{col}_today' for col in diet_travel_cols]].max(axis=1)
    df['consecutive_trigger_days'] = any_diet_travel.groupby(df['patient_id']).apply(
        lambda x: count_consecutive(x, 1)).reset_index(0, drop=True)

    # --- Physical Activity ---
    df['exercise_today'] = ((df['vigorous_exercise_min'] > 0) | (df['moderate_exercise_min'] > 0)).astype(int)
    df['no_exercise_today'] = df['no_exercise']
    df['consecutive_exercise_days'] = grp['exercise_today'].apply(lambda x: count_consecutive(x, 1)).reset_index(0,
                                                                                                                 drop=True)
    df['consecutive_sedentary_days'] = grp['exercise_today'].apply(lambda x: count_consecutive(x, 0)).reset_index(0,
                                                                                                                  drop=True)
    df['exercise_days_7day'] = grp['exercise_today'].rolling(7, min_periods=1).sum().reset_index(0, drop=True)

    # --- Other Triggers & Hormonal ---
    other_triggers = ['physical_fatigue', 'emotional_changes', 'noise', 'specific_smells', 'menstruation', 'ovulation']
    for col in other_triggers:
        df[f'{col}_today'] = df[col]

    # Context
    df['dow'] = df['date'].dt.dayofweek
    
    global DATE_70, DATE_85

    # Train / Val / Test Split (70/15/15 Chronological)
    unique_dates = df['date'].sort_values().unique()
    DATE_70 = unique_dates[int(len(unique_dates) * 0.70)]
    DATE_85 = unique_dates[int(len(unique_dates) * 0.85)]

    conditions = [
        df['date'] < DATE_70,
        (df['date'] >= DATE_70) & (df['date'] < DATE_85)
    ]
    choices = ['train', 'val']
    df['split'] = np.select(conditions, choices, default='test')

    # 8. Drop structural columns and base trigger columns
    structural_drops = ['headache_ongoing', 'severity_category', 'severity_vas', 'headache_free']
    base_triggers = ['stress', 'oversleeping', 'lack_of_sleep', 'weather_change', 'irregular_meals',
                     'overeating', 'excessive_caffeine', 'alcohol', 'travel', 'no_exercise',
                     'physical_fatigue', 'emotional_changes', 'noise', 'specific_smells',
                     'menstruation', 'ovulation']

    cols_to_drop = structural_drops + base_triggers
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    return df

"""
Step 3 — Stage 5 Supplement (Sheet 2 Disability)

Sheet 2 is processed separately into separated Train, Val, and Test Parquet files and is not included in the Stage 0–4 feature matrix. It is joined to the daily diary by patient ID and date for Stage 5 experiments only.

Train/Val/Test Split Alignment:
To ensure zero data leakage and exact temporal alignment with the daily diary features, the disability dataset is split into chronologically identical Train (70%), Validation (15%), and Test (15%) sets using the exact date cutoffs established in Step 2. Orphan patients without daily diary logs are filtered out. Outputs are physically separated into `train_disability.parquet`, etc.

Reading procedure for Sheet 2:

1. Open `data/SHD-Dataset.xls` with `xlrd<2.0`, sheet `headache diary 1099`.
2. Read with `header=[1, 2]` (data begins at row 3); forward-fill the group header.
3. Combine into single-level headers using the `{group} — {subheader}` convention.
4. Filter rows: keep only rows where `등록번호` matches a valid patient identifier.
5. Apply the column mapping below; drop all unlisted columns.

Korean → English column mapping for Sheet 2:

| Korean header (verbatim) | Output column | Notes |
|--------------------------|---------------|-------|
| 번호 | entry_id | identifier; matches Sheet 3 `번호` for join |
| 등록번호 | patient_id | uppercased |
| 날짜 | date | parsed via `format='%Y%m%d'` |
| 정도 | severity_category | mild/moderate/severe |
| Pain intensity (VAS) | severity_vas | 0–10 |
| 동반증상 — 편두통유무 | migraine_flag | ICHD migraine classification |
| 유발요인 — 유발요인 수 | trigger_count | integer trigger count |
| 지속시간(분) | headache_duration_min | duration in minutes |
| 구급약 사용1 — 장애 | disability_any | overall disability flag (col 50) |
| 학교 또는 직장 — 결근하거나 등교하지 못하였다 | disability_work_severe | severe work/school disability (col 51) |
| 학교 또는 직장 — 작업또는 학업능률이 절반 이하로 감소하였다 | disability_work_moderate | moderate work/school disability (col 52) |
| 학교 또는 직장 — 합계 | disability_work_affected | work/school subtotal (col 53) |
| 집안에서 — 가사일을 전혀 할 수 없었다 | disability_housework_severe | severe housework disability (col 54) |
| 집안에서 — 가사의 능률이 절반 이하로 감소하였다 | disability_housework_moderate | moderate housework disability (col 55) |
| 집안에서 — 합계 | disability_housework_affected | housework subtotal (col 56) |
| 모임/여가 활동 — 예정이 있었으나 참여 할 수 없었다 | disability_social | social/leisure missed (col 57) |
| All 18 trigger flag columns (cols 60–84) | trigger flags | retain with same names as in `translated.parquet` for join |

Columns retained:

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
"""
def process_disability_sheet(raw_df):
    """Step 3 — Stage 5 Supplement (Sheet 2 Disability)"""
    df = raw_df.copy()
    df = _flatten_multiindex_columns(df)

    col_map = {
        '번호': 'entry_id',
        '등록번호': 'patient_id',
        '날짜': 'date',
        '정도': 'severity_category',
        'Pain intensity (VAS)': 'severity_vas',
        '동반증상 — 편두통유무': 'migraine_flag',
        '유발요인 — 유발요인 수': 'trigger_count',
        '지속시간(분)': 'headache_duration_min',
        '구급약 사용1 — 장애': 'disability_any',
        '학교 또는 직장 — 결근하거나 등교하지 못하였다': 'disability_work_severe',
        '학교 또는 직장 — 작업또는 학업능률이 절반 이하로 감소하였다': 'disability_work_moderate',
        '학교 또는 직장 — 합계': 'disability_work_affected',
        '집안에서 — 가사일을 전혀 할 수 없었다': 'disability_housework_severe',
        '집안에서 — 가사의 능률이 절반 이하로 감소하였다': 'disability_housework_moderate',
        '집안에서 — 합계': 'disability_housework_affected',
        '모임/여가 활동 — 예정이 있었으나 참여 할 수 없었다': 'disability_social',

        # 18 Trigger flag columns mapping identical to translated.parquet
        '내인적 요인 — 스트레스': 'stress',
        '내인적 요인 — 수면과다': 'oversleeping',
        '내인적 요인 — 수면부족': 'lack_of_sleep',
        '내인적 요인 — 운동 안하기': 'no_exercise',
        '내인적 요인 — 육체적 피로': 'physical_fatigue',
        '내인적 요인 — 생리주기 : 월경기': 'menstruation',
        '내인적 요인 — 생리주기 : 배란기': 'ovulation',
        '내인적 요인 — 과도한 감정변화': 'emotional_changes',
        '외부적 요인 — 날씨/온도 변화': 'weather_change',
        '외부적 요인 — 소음': 'noise',
        '외부적 요인 — 특정한 냄새(화장품 향수 등)': 'specific_smells',
        '기타 — 과도한 음주': 'alcohol',
        '기타 — 불규칙한 식사(공복 등)': 'irregular_meals',
        '기타 — 과식': 'overeating',
        '기타 — 과도한 카페인 음료': 'excessive_caffeine',
        '기타 — 여행': 'travel'
    }

    keep_cols = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(keep_cols.keys())].rename(columns=keep_cols)

    # Filter valid patient identifiers
    df = df.dropna(subset=['patient_id'])
    df = df[~df['patient_id'].astype(str).str.contains('합계|total', case=False, na=False)]

    # Apply standard formats
    df['patient_id'] = df['patient_id'].astype(str).str.upper().str.strip()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'].astype(str).str.extract(r'(\d{8})')[0], format='%Y%m%d')

    return df


def print_data_insights(raw_df, translated_df,
                                   train_engineered, val_engineered, test_engineered,
                                   train_disability, val_disability, test_disability):
    """
    Generates a rigorous scientific data profile for all 8 pipeline datasets,
    adding distributions, statistical moments, and time-series integrity checks.
    """

    datasets = {
        "1. RAW DIARY DATA": raw_df,
        "2. TRANSLATED DATA": translated_df,
        "3A. ENGINEERED DATA (TRAIN)": train_engineered,
        "3B. ENGINEERED DATA (VAL)": val_engineered,
        "3C. ENGINEERED DATA (TEST)": test_engineered,
        "4A. DISABILITY SUPPLEMENT (TRAIN)": train_disability,
        "4B. DISABILITY SUPPLEMENT (VAL)": val_disability,
        "4C. DISABILITY SUPPLEMENT (TEST)": test_disability
    }

    for name, df in datasets.items():
        total_rows = len(df)
        total_cols = len(df.columns)

        print("=" * 120)
        print(f" {name}")
        print(f" SHAPE: {total_rows:,} rows | {total_cols:,} columns")

        # Prevent crash if a split is completely empty
        if total_rows == 0:
            print("-" * 120)
            print(" [EMPTY DATAFRAME - Check split logic]")
            print("\n")
            continue

        # --- Time-Series & Cohort Sanity Checks (if applicable) ---
        if 'patient_id' in df.columns and 'date' in df.columns:
            n_patients = df['patient_id'].nunique()
            date_min, date_max = df['date'].min(), df['date'].max()
            days_per_patient = df.groupby('patient_id').size()
            dupes = df.duplicated(subset=['patient_id', 'date']).sum()

            print("-" * 120)
            print(f" COHORT INTEGRITY:")
            if pd.notna(date_min) and pd.notna(date_max):
                print(
                    f" • Patients: {n_patients} | Date Range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
            print(
                f" • Days per patient -> Min: {days_per_patient.min()} | Median: {days_per_patient.median():.1f} | Max: {days_per_patient.max()}")
            print(f" • Duplicate (Patient, Date) rows: {dupes} " + ("(WARNING!)" if dupes > 0 else "(Clean)"))

        print("-" * 120)

        # --- Column-by-Column Profiling ---
        for col in df.columns:
            # Flatten MultiIndex if raw_df
            if isinstance(col, tuple):
                col_name = " — ".join([str(c) for c in col if pd.notna(c) and not str(c).startswith('Unnamed:')])
            else:
                col_name = str(col)

            dtype = str(df[col].dtype)
            null_count = df[col].isna().sum()
            valid_count = total_rows - null_count
            null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
            unique_count = df[col].nunique()

            # Base string
            info_str = f"• {col_name[:35]:<35} | {dtype:<10} | Nulls: {null_count:>5} ({null_pct:>5.1f}%) | Unq: {unique_count:>4}"

            # 1. Distribution Check for Binary / Categorical (<= 5 unique values)
            if unique_count > 0 and unique_count <= 5 and not pd.api.types.is_float_dtype(df[col]):
                val_counts = df[col].value_counts(dropna=False).to_dict()
                # Format the dictionary nicely
                dist_str = ", ".join([f"{k}: {v}" for k, v in val_counts.items()])
                info_str += f" | Dist: [{dist_str}]"

            # 2. Statistical Check for Continuous / High-Cardinality Numeric
            elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                std_val = df[col].std()

                if pd.notna(min_val):
                    info_str += f" | Min: {min_val:>6.1f} | Max: {max_val:>6.1f} | Mean: {mean_val:>6.1f} | Std: {std_val:>6.1f}"

            print(info_str)

        print("\n")

# Read in SHD-Dataset.xls Sheet 3 (daily diary data)
# Note: sheet_name=2 is the 3rd sheet (0-indexed). header=[1, 2] captures the multi-row headers.
raw_df = pd.read_excel('SHD-Dataset.xls', sheet_name=2, header=[1, 2])

# Step 1 — Translation -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
translated_df = translate_sheet3(raw_df)
translated_df.to_parquet('translated.parquet', index=False)

print("Saved: translated.parquet")

# Step 2 — Engineering -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
engineered_df = engineer_features(translated_df)

# Split and save Engineered Data
train_engineered = engineered_df[engineered_df['split'] == 'train'].drop(columns=['split'])
val_engineered = engineered_df[engineered_df['split'] == 'val'].drop(columns=['split'])
test_engineered = engineered_df[engineered_df['split'] == 'test'].drop(columns=['split'])

train_engineered.to_parquet('train_engineered.parquet', index=False)
val_engineered.to_parquet('val_engineered.parquet', index=False)
test_engineered.to_parquet('test_engineered.parquet', index=False)

print(f"Engineered -> Train: {len(train_engineered)} | Val: {len(val_engineered)} | Test: {len(test_engineered)}")

# Step 3 — Stage 5 Supplement (Sheet 2 Disability) -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Note: sheet_name=1 is the 2nd sheet (0-indexed).
raw_disability_df = pd.read_excel('SHD-Dataset.xls', sheet_name=1, header=[1, 2])
disability_df = process_disability_sheet(raw_disability_df)

# Apply the same train/val/test split as engineered_df for consistency in Stage 5 experiments
disability_df['split'] = np.select(
    [
        disability_df['date'] < DATE_70,
        (disability_df['date'] >= DATE_70) & (disability_df['date'] < DATE_85)
    ],
    ['train', 'val'],
    default='test'
)

train_disability = disability_df[disability_df['split'] == 'train'].drop(columns=['split'])
val_disability = disability_df[disability_df['split'] == 'val'].drop(columns=['split'])
test_disability = disability_df[disability_df['split'] == 'test'].drop(columns=['split'])

train_disability.to_parquet('train_disability.parquet', index=False)
val_disability.to_parquet('val_disability.parquet', index=False)
test_disability.to_parquet('test_disability.parquet', index=False)

print(f"Disability -> Train: {len(train_disability)} | Val: {len(val_disability)} | Test: {len(test_disability)}")

print("\n   Pipeline complete. Parquet files saved on same level.  \n") # -- -- -- -- -- --

print_data_insights(raw_df, translated_df,
       train_engineered, val_engineered, test_engineered,
       train_disability, val_disability, test_disability)
