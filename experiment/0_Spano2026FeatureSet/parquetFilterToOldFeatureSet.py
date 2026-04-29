import pandas as pd
from typing import Optional


def remove_non_spano_features(input_parquet_path: str, output_parquet_path: Optional[str] = None) -> pd.DataFrame:
    """
    Reads a feature-engineered Parquet file and removes columns that the
    current benchmark uses, but Spano (2026) did not use at all.

    Note: Weather features are NOT removed here because Spano did attempt
    to use them (unsuccessfully, due to a "Wheater" typo).

    Args:
        input_parquet_path (str): Path to the input Parquet file.
        output_parquet_path (str, optional): Path to save the cleaned Parquet file.

    Returns:
        pd.DataFrame: The DataFrame with the non-Spano features removed.
    """
    # 1. Load the parquet file
    df = pd.read_parquet(input_parquet_path)

    # 2. Define the exact columns Spano did not use at all, but the benchmark does.
    # (Mappings based on standard dataset naming conventions from your docs)
    columns_to_remove = [
        "physical_fatigue_today",
        "emotional_changes_today",
        "noise_today",
        "specific_smells_today",
        "vigorous_exercise_min",
        "moderate_exercise_min",
        "preventive_medication"
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