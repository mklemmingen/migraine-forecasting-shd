"""
engineer.py - Step 2: Feature engineering on the translated diary DataFrame.

Builds all 47 temporal/rolling features. Does NOT apply any split - returns the
full engineered DataFrame. The caller (a run_pipeline_*.py script) applies the
appropriate split strategy from data/pipeline/splits/.

Rolling window edge handling: all rolling features use min_periods=1 so partial
windows at the start of each patient's series compute over available days, avoiding
dropping the first 6 days per patient.

Feature groups and counts:
  Migraine history (5), Stress (3), Sleep (7), Weather (5),
  Dietary & Travel (6), Physical Activity (6), Other triggers (6),
  Hormonal (2), Preventive medication (1), Context / dow (1) , (2) days since last record and recording gap,
  (5) low-count triggers retained for the base feature set (Park et al. excluded them from sub-analysis)
  → 47 features total

Excluded at engineering: Structural columns headache_ongoing, severity_category,
severity_vas, headache_free dropped.
"""
import pandas as pd


def engineer_features(
    translated_df: pd.DataFrame,
    target_mode: str = "headache",
    disability_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all temporal/rolling features from the translated diary DataFrame.

    Args:
        target_mode: "headache" (any headache, default) or "migraine"
            (ICHD-3 migraine only, requires disability_df).
        disability_df: Sheet 2 DataFrame with migraine_flag column.
            Required when target_mode="migraine".

    Returns the full engineered DataFrame including migraine_today and migraine_target.
    No 'split' column is added here.
    """
    df = translated_df.sort_values(['patient_id', 'date']).copy()

    def count_consecutive(s, val):
        is_val = (s == val).astype(int)
        blocks = (is_val == 0).cumsum()
        return is_val.groupby(blocks).cumsum()

    # --- Target construction (mode-dependent) ---
    if target_mode == "headache":
        df['migraine_today'] = (df['headache_free'] == 0).astype(int)
    elif target_mode == "migraine":
        if disability_df is None:
            raise ValueError("disability_df required for target_mode='migraine'")
        # Any headache day first (needed for structural columns)
        df['_any_headache'] = (df['headache_free'] == 0).astype(int)
        # Join migraine_flag from Sheet 2
        migraine_days = disability_df[['patient_id', 'date', 'migraine_flag']].copy()
        migraine_days['migraine_flag'] = migraine_days['migraine_flag'].astype(int)
        df = df.merge(migraine_days, on=['patient_id', 'date'], how='left')
        df['migraine_today'] = df['migraine_flag'].fillna(0).astype(int)
        df = df.drop(columns=['migraine_flag', '_any_headache'])
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    df['migraine_target'] = df.groupby('patient_id')['migraine_today'].shift(-1)
    df = df.dropna(subset=['migraine_target']).copy()

    grp = df.groupby('patient_id')

    # --- Migraine history ---
    df['migraine_yesterday'] = grp['migraine_today'].shift(1).fillna(0).astype(int)
    df['migraine_rate_last3'] = grp['migraine_today'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df['migraine_rate_last7'] = grp['migraine_today'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    df['headache_free_streak'] = grp['migraine_today'].apply(lambda x: count_consecutive(x, 0)).reset_index(0, drop=True)

    last_migraine = df['date'].where(df['migraine_today'] == 1)
    last_migraine_ffill = last_migraine.groupby(df['patient_id']).ffill()
    df['days_since_last_migraine'] = (df['date'] - last_migraine_ffill).dt.days.fillna(61).astype(int)

    # --- Stress ---
    df['stress_today'] = df['stress']
    stress_yesterday = grp['stress_today'].shift(1).fillna(0).astype(int)
    df['stress_drop_today'] = ((df['stress_today'] == 0) & (stress_yesterday == 1)).astype(int)
    df['consecutive_stress_days'] = grp['stress_today'].apply(lambda x: count_consecutive(x, 1)).reset_index(0, drop=True)

    # --- Sleep ---
    df['lack_of_sleep_today'] = df['lack_of_sleep']
    df['oversleeping_today'] = df['oversleeping']
    df['any_sleep_issue_today'] = (df['lack_of_sleep_today'] | df['oversleeping_today']).astype(int)
    df['sleep_debt_3day'] = grp['lack_of_sleep_today'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
    df['sleep_disruption_today'] = (df['any_sleep_issue_today'] & df['migraine_yesterday']).astype(int)
    df['sleep_variability_7day'] = (grp['any_sleep_issue_today']
                                    .rolling(7, min_periods=1).std()
                                    .reset_index(0, drop=True)
                                    .fillna(0))
    is_weekend = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['recent_weekend_sleep_issues'] = ((df['any_sleep_issue_today'] & is_weekend)
                                         .groupby(df['patient_id'])
                                         .rolling(7, min_periods=1).sum()
                                         .reset_index(0, drop=True))
    df['exercise_as_trigger_today'] = df['exercise_as_trigger']
    df['sunlight_today'] = df['sunlight']
    df['inappropriate_lighting_today'] = df['inappropriate_lighting']
    df['excessive_smoking_today'] = df['excessive_smoking']
    df['cheese_chocolate_today'] = df['cheese_chocolate']

    # --- Weather ---
    df['weather_change_today'] = df['weather_change']
    df['consecutive_weather_changes'] = grp['weather_change_today'].apply(lambda x: count_consecutive(x, 1)).reset_index(0, drop=True)
    df['weather_instability_3day'] = grp['weather_change_today'].rolling(3, min_periods=1).sum().reset_index(0, drop=True)
    df['weather_change_yesterday'] = grp['weather_change_today'].shift(1).fillna(0).astype(int)
    df['weather_headache_interaction'] = (df['weather_change_today'] & df['migraine_yesterday']).astype(int)

    # --- Dietary & Travel ---
    diet_travel_cols = ['irregular_meals', 'overeating', 'excessive_caffeine', 'alcohol', 'travel']
    for col in diet_travel_cols:
        df[f'{col}_today'] = df[col]
    any_diet_travel = df[[f'{col}_today' for col in diet_travel_cols]].max(axis=1)
    df['consecutive_trigger_days'] = (any_diet_travel
                                      .groupby(df['patient_id'])
                                      .apply(lambda x: count_consecutive(x, 1))
                                      .reset_index(0, drop=True))

    # --- Physical Activity ---
    df['exercise_today'] = ((df['vigorous_exercise_min'] > 0) | (df['moderate_exercise_min'] > 0)).astype(int)
    df['no_exercise_today'] = df['no_exercise']
    df['consecutive_exercise_days'] = grp['exercise_today'].apply(lambda x: count_consecutive(x, 1)).reset_index(0, drop=True)
    df['consecutive_sedentary_days'] = grp['exercise_today'].apply(lambda x: count_consecutive(x, 0)).reset_index(0, drop=True)
    df['exercise_days_7day'] = grp['exercise_today'].rolling(7, min_periods=1).sum().reset_index(0, drop=True)

    # --- Gap awareness ---
    df["days_since_last_record"] = (
        grp["date"].diff().dt.days.fillna(1).astype(int)
    )
    df["recording_gap_flag"] = (df["days_since_last_record"] > 1).astype(int)

    # --- Other triggers & Hormonal ---
    for col in ['physical_fatigue', 'emotional_changes', 'noise', 'specific_smells', 'menstruation', 'ovulation']:
        df[f'{col}_today'] = df[col]

    # --- Context ---
    df['dow'] = df['date'].dt.dayofweek

    # Drop structural and base trigger columns
    structural_drops = ['headache_ongoing', 'severity_category', 'severity_vas', 'headache_free']
    base_triggers = [
        'stress', 'oversleeping', 'lack_of_sleep', 'weather_change',
        'irregular_meals', 'overeating', 'excessive_caffeine', 'alcohol', 'travel',
        'no_exercise', 'physical_fatigue', 'emotional_changes', 'noise',
        'specific_smells', 'menstruation', 'ovulation',
        # raw inputs whose *_today encodings are produced above (lines 102-106)
        'exercise_as_trigger', 'sunlight', 'inappropriate_lighting',
        'excessive_smoking', 'cheese_chocolate',
    ]
    df = df.drop(columns=[c for c in structural_drops + base_triggers if c in df.columns], errors='ignore')

    return df
