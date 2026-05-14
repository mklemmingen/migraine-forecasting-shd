"""filter_to_spano_features.py - Replicate Spano (2026) column selection.

Reads a feature-engineered Parquet file and returns a DataFrame whose
columns match the per-domain features Spano (2026) used in his
bachelor-thesis pipeline (see ``headfree-backend/features/*.py``).
Anything not in the whitelist is dropped.

Spano (2026) feature reference: headfree-backend/features/*.py per
domain plus headfree-backend/build_features.py for the orchestrator
``add_history_features`` step. Spano's full feature set is ~32
per-domain features plus 6 history features.

Notes on selected design points:

- ``cheese_chocolate_today`` is **kept** - it is Spano's
  ``trigger_foods_today`` under SHD's column name. Same boolean values,
  different name. The filter strategy is by-column-presence not by
  rename, so the SHD label flows through.

- ``sleep_disruption_today`` is **omitted** because the SHD definition
  ``(any_sleep_issue & migraine_yesterday)`` is target-derived while
  Spano's definition uses yesterday's sleep flag. Same name, different
  semantics; omitting is more honest than passing forward a column
  whose values do not match Spano's pipeline.

- Several columns Spano had as engineered features are NOT in the
  current engineered parquets (see SPANO_FEATURES_MISSING_FROM_SHD
  below). A filter cannot synthesise missing columns; those gaps are
  documented in code so the methodology section can disclose them.

Plugs into ``_dataRead.read.load_and_prep_data`` via the ``loader=``
parameter.
"""
from pathlib import Path

import pandas as pd

from _dataRead._select_columns import select_columns


# Spano-engineered features that data/pipeline/engineer.py does NOT
# produce. A filter cannot synthesise them - listed here so the gap
# is documented in code.
SPANO_FEATURES_MISSING_FROM_SHD = (
    "exercise_consistency_7day",   # rolling: >=3 of last 7 days exercised
    "exercise_disruption",         # |today - trailing 7-day mean| > 0.5
    "smoking_withdrawal_today",    # 3-day excessive smoking -> 0 today
    "travel_exercise_conflict",    # travel_today AND no exercise
    "weather_changes_3day_count",  # raw count (Spano also keeps the boolean)
)


# Whitelist of columns the engineered parquet must provide for the
# Spano-feature variant. Anything not here is dropped; anything here
# but missing raises a clear error.
_SPANO_REQUIRED_COLUMNS = (
    # Structural
    "patient_id",
    "date",
    "migraine_target",
    # Stress
    "stress_today",
    "stress_drop_today",
    "consecutive_stress_days",
    # Sleep (Spano-relevant subset; sleep_disruption_today omitted per
    # docstring)
    "lack_of_sleep_today",
    "oversleeping_today",
    "any_sleep_issue_today",
    "sleep_debt_3day",
    "sleep_variability_7day",
    "recent_weekend_sleep_issues",
    # Weather (Spano-relevant subset; weather_change_yesterday and
    # weather_headache_interaction are SHD-specific lag/interaction
    # variants Spano did not use)
    "weather_change_today",
    "consecutive_weather_changes",
    "weather_instability_3day",
    # Dietary / travel
    "irregular_meals_today",
    "overeating_today",
    "excessive_caffeine_today",
    "alcohol_today",
    "travel_today",
    "consecutive_trigger_days",
    # Physical activity (Spano's exercise_today boolean form; the SHD
    # raw vigorous_exercise_min / moderate_exercise_min and the
    # no_exercise_today flag are NOT Spano-features)
    "exercise_today",
    "consecutive_exercise_days",
    "consecutive_sedentary_days",
    "exercise_days_7day",
    # Hormonal
    "menstruation_today",
    "ovulation_today",
    # Calendar
    "dow",
    # Migraine history (Spano had migraine_yesterday, migraine_rate_last3,
    # migraine_rate_last7, days_since_last_migraine; headache_free_streak
    # is SHD-specific and omitted)
    "migraine_yesterday",
    "migraine_rate_last3",
    "migraine_rate_last7",
    "days_since_last_migraine",
    # Low-count triggers Spano did capture
    "excessive_smoking_today",
    "cheese_chocolate_today",
)

_SPANO_OPTIONAL_COLUMNS = (
    "entry_id",
    "cv_fold",
    # See filter_to_no_rolling_features.py note: migraine_today is
    # currently dropped at the split step; declared optional in case
    # the pipeline starts retaining it.
    "migraine_today",
)


def select_spano_features(parquet_path: str | Path) -> pd.DataFrame:
    """Return a DataFrame with only Spano (2026)-matching columns.

    Spano-features that the current engineering pipeline does NOT
    produce are listed in ``SPANO_FEATURES_MISSING_FROM_SHD`` for the
    methodology section to disclose.
    """
    df = select_columns(
        parquet_path,
        required=_SPANO_REQUIRED_COLUMNS,
        optional=_SPANO_OPTIONAL_COLUMNS,
    )
    structural_in_df = sum(
        c in df.columns
        for c in ("patient_id", "date", "entry_id", "cv_fold",
                  "migraine_target", "migraine_today")
    )
    n_features = df.shape[1] - structural_in_df
    print(
        f"spano feature set: {df.shape[0]} rows x {n_features} features "
        f"(per-domain Spano (2026) columns; "
        f"{len(SPANO_FEATURES_MISSING_FROM_SHD)} Spano features not "
        f"reproducible from current engineering)."
    )
    return df
