"""
filter_to_no_rolling_features.py - Drop temporal aggregation columns.

Reads a feature-engineered Parquet file and removes all rolling, lagged,
interaction, streak, and gap-awareness features built by
data/pipeline/engineer.py - leaving only same-day trigger flags, calendar
context (dow), and current-day measurements.

Purpose: scientific decomposition. Comparing a model trained on full features
against the same model trained on the no-rolling subset isolates how much
predictive signal comes from temporal autocorrelation (rolling target history,
streaks, lagged interactions) versus from today's observed triggers alone -
the question Park et al. (2016) studied for same-day associations.

Mirrors the API of filter_to_spano_features.remove_non_spano_features so it
plugs into _dataRead.read.load_and_prep_data via the `loader=` parameter.
"""
from typing import Optional

import pandas as pd


# Columns built by engineer.py from rolling, shifted, streak-counted, or
# interaction-with-lagged-target operations. Removing them leaves a feature
# matrix where every row's predictors describe only the current diary day.
ROLLING_AND_LAG_COLUMNS = [
    # --- Target-derived lag/rolling history -------------------------------
    "migraine_yesterday",
    "migraine_rate_last3",
    "migraine_rate_last7",
    "headache_free_streak",
    "days_since_last_migraine",
    # --- Stress lag/streak ------------------------------------------------
    "stress_drop_today",          # (stress_today == 0) & (stress_yesterday == 1)
    "consecutive_stress_days",
    # --- Sleep rolling/interaction ----------------------------------------
    "sleep_debt_3day",
    "sleep_disruption_today",     # any_sleep_issue & migraine_yesterday
    "sleep_variability_7day",
    "recent_weekend_sleep_issues",
    # --- Weather rolling/lag/interaction ----------------------------------
    "consecutive_weather_changes",
    "weather_instability_3day",
    "weather_change_yesterday",
    "weather_headache_interaction",   # weather_change & migraine_yesterday
    # --- Diet/travel streak -----------------------------------------------
    "consecutive_trigger_days",
    # --- Exercise rolling/streak ------------------------------------------
    "consecutive_exercise_days",
    "consecutive_sedentary_days",
    "exercise_days_7day",
    # --- Gap awareness (target-related - meaningless without rolling) -----
    "days_since_last_record",
    "recording_gap_flag",
]


def remove_rolling_features(
    input_parquet_path: str,
    output_parquet_path: Optional[str] = None,
) -> pd.DataFrame:
    """Read an engineered Parquet and drop all temporal-aggregation columns.

    Args:
        input_parquet_path: Path to the engineered diary parquet.
        output_parquet_path: Optional path to write the filtered parquet.

    Returns:
        DataFrame with only same-day trigger flags, dow, identifiers, and target.
    """
    df = pd.read_parquet(input_parquet_path)

    # Tolerate absence - schema may evolve and not all engineering passes
    # produce every column.
    cols_to_drop = [c for c in ROLLING_AND_LAG_COLUMNS if c in df.columns]

    if cols_to_drop:
        print(f"Removing {len(cols_to_drop)} rolling/lag/interaction features:")
        for col in sorted(cols_to_drop):
            print(f" - {col}")
        df_cleaned = df.drop(columns=cols_to_drop)
    else:
        print("No rolling/lag/interaction features found in this file.")
        df_cleaned = df.copy()

    if output_parquet_path:
        df_cleaned.to_parquet(output_parquet_path, index=False)
        print(f"\nFiltered dataset saved to: {output_parquet_path}")

    return df_cleaned
