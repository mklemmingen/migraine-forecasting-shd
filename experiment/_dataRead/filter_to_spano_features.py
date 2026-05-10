import pandas as pd
from typing import Optional


# Spano-engineered features that data/pipeline/engineer.py does NOT produce.
# These would need to be added to engineer.py for a 1:1 Spano-faithful feature
# set on the SHD dataset. The filter cannot synthesise them — it only drops
# columns that exist. Listed here so the gap is documented in code.
SPANO_FEATURES_MISSING_FROM_SHD = [
    "exercise_consistency_7day",   # rolling: ≥3 of last 7 days exercised
    "exercise_disruption",         # |today - trailing 7-day mean| > 0.5
    "smoking_withdrawal_today",    # 3-day excessive smoking → 0 today
    "travel_exercise_conflict",    # travel_today AND no exercise
    "weather_changes_3day_count",  # raw count (Spano keeps the boolean form too)
]


def remove_non_spano_features(input_parquet_path: str, output_parquet_path: Optional[str] = None) -> pd.DataFrame:
    """
    Reads a feature-engineered Parquet file and removes columns that the
    current benchmark engineers but Spano (2026) did not use.

    Reference: headfree-backend/features/*.py (per-domain feature modules) and
    headfree-backend/build_features.py (orchestrator with add_history_features).
    Spano's full feature set ≈ 32 per-domain features + 6 history features.

    See `SPANO_FEATURES_MISSING_FROM_SHD` above for the 5 Spano features that
    engineer.py does NOT produce — we cannot recover them with a filter.

    Notes
    -----
    Weather features are NOT all removed: Spano had weather_change_today,
    consecutive_weather_changes, and weather_instability_3day. Removed are the
    SHD-specific weather_change_yesterday (lag) and weather_headache_interaction
    (target-derived).

    sleep_disruption_today is dropped despite both pipelines having a column of
    that name: SHD computes it as (any_sleep_issue & migraine_yesterday) — i.e.
    target-derived — while Spano computes it from yesterday's sleep flag only.
    Same name, different semantics; dropping is more honest than passing
    forward a column whose values do not match Spano's pipeline.

    cheese_chocolate_today is KEPT — it is Spano's "trigger_foods_today" under
    SHD's column name. Same values, different name; the filter strategy is
    by-column-presence, not by-rename.

    Args:
        input_parquet_path (str): Path to the input Parquet file.
        output_parquet_path (str, optional): Path to save the cleaned Parquet file.

    Returns:
        pd.DataFrame: The DataFrame with the non-Spano features removed.
    """
    # 1. Load the parquet file
    df = pd.read_parquet(input_parquet_path)

    # 2. Columns SHD engineers but Spano (2026) did not use.
    columns_to_remove = [
        # Other-trigger features Spano's diary app didn't capture
        "physical_fatigue_today",
        "emotional_changes_today",
        "noise_today",
        "specific_smells_today",
        "exercise_as_trigger_today",
        "sunlight_today",
        "inappropriate_lighting_today",
        # Raw exercise input columns (Spano had Exercise/No-Exercise booleans, not minutes)
        "vigorous_exercise_min",
        "moderate_exercise_min",
        "no_exercise_today",            # Spano resolved into a single exercise_today flag
        # Medication
        "preventive_medication",
        # Migraine-history features Spano didn't compute (he has migraine_yesterday/_rate_last3/_last7/days_since_last_migraine, not these)
        "headache_free_streak",
        # Weather features specific to SHD (lag + target interaction)
        "weather_change_yesterday",
        "weather_headache_interaction",
        # Gap awareness — Spano fills missing days, doesn't track gaps
        "days_since_last_record",
        "recording_gap_flag",
        # Same name, different semantics: SHD uses (sleep & migraine_yesterday); Spano uses (sleep & sleep_yesterday)
        "sleep_disruption_today",
    ]

    # 3. Identify any disability outcome columns (introduced in Stage 5, not used by Spano)
    disability_keywords = ["disability", "midas", "outcome_domain"]
    disability_cols = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in disability_keywords)
    ]

    # Combine all targets to drop
    all_targets_to_drop = set(columns_to_remove + disability_cols)

    # 4. Filter to only those columns that actually exist in the current DataFrame
    # (prevents KeyError if some features haven't been engineered into this specific split yet)
    cols_to_drop = [col for col in all_targets_to_drop if col in df.columns]

    if cols_to_drop:
        print(f"Removing {len(cols_to_drop)} benchmark-exclusive features:")
        for col in sorted(cols_to_drop):
            print(f" - {col}")

        # Drop the columns
        df_cleaned = df.drop(columns=cols_to_drop)
    else:
        print("No benchmark-exclusive non-Spano features found in this file.")
        df_cleaned = df.copy()

    # 5. Save to disk if an output path is provided
    if output_parquet_path:
        df_cleaned.to_parquet(output_parquet_path, index=False)
        print(f"\nCleaned dataset saved to: {output_parquet_path}")

    return df_cleaned