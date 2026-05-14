"""filter_to_no_rolling_features.py - Same-day-only feature subset.

Reads a feature-engineered Parquet file and returns a DataFrame that
contains **only** same-day trigger flags, calendar context (``dow``),
identifiers, and the target. All rolling, lagged, interaction, streak,
and gap-awareness features built by ``data/pipeline/engineer.py`` are
omitted by NOT being in the whitelist - so if the engineering pipeline
gains new columns, they default to "not included" rather than silently
leaking through into every leaf.

Purpose: scientific decomposition. Comparing a model trained on full
features against the same model trained on the same-day-only subset
isolates how much predictive signal comes from temporal autocorrelation
(rolling target history, streaks, lagged interactions) versus from
today's observed triggers alone - the question Park et al. (2016)
studied for same-day associations.

Plugs into ``_dataRead.read.load_and_prep_data`` via the ``loader=``
parameter.
"""
from pathlib import Path

import pandas as pd

from _dataRead._select_columns import select_columns


# Same-day flags + raw inputs + calendar context that engineer.py emits.
# Anything not in this list (rolling / lag / streak / gap / interaction)
# is intentionally dropped.
_NON_ROLLING_REQUIRED_COLUMNS = (
    # Structural
    "patient_id",
    "date",
    "migraine_target",
    # Stress
    "stress_today",
    # Sleep (same-day only; rolling/lag/interaction members live in the
    # full_features set)
    "lack_of_sleep_today",
    "oversleeping_today",
    "any_sleep_issue_today",
    # Weather (same-day only)
    "weather_change_today",
    # Dietary / travel
    "irregular_meals_today",
    "overeating_today",
    "excessive_caffeine_today",
    "alcohol_today",
    "travel_today",
    # Physical activity (same-day flags + raw minute inputs)
    "exercise_today",
    "no_exercise_today",
    "vigorous_exercise_min",
    "moderate_exercise_min",
    # Other triggers (same-day)
    "physical_fatigue_today",
    "emotional_changes_today",
    "noise_today",
    "specific_smells_today",
    # Hormonal
    "menstruation_today",
    "ovulation_today",
    # Calendar
    "dow",
    # Low-count triggers retained for the base feature set (Park et al.
    # excluded them from sub-analysis, see docs/dataset.md)
    "exercise_as_trigger_today",
    "sunlight_today",
    "inappropriate_lighting_today",
    "excessive_smoking_today",
    "cheese_chocolate_today",
)

_NON_ROLLING_OPTIONAL_COLUMNS = (
    "entry_id",
    "cv_fold",
    # ``migraine_today`` is currently dropped by the upstream split step
    # (engineer.py emits it but the diary_<split>.parquet writers do not
    # carry it through). Listed as optional so the filter still works if
    # a future pipeline pass starts retaining it.
    "migraine_today",
)


def select_non_rolling_features(parquet_path: str | Path) -> pd.DataFrame:
    """Return a DataFrame with only same-day flags and identifiers.

    See module docstring for the rationale; the whitelist is the
    ``_NON_ROLLING_REQUIRED_COLUMNS`` tuple.
    """
    df = select_columns(
        parquet_path,
        required=_NON_ROLLING_REQUIRED_COLUMNS,
        optional=_NON_ROLLING_OPTIONAL_COLUMNS,
    )
    structural_in_df = sum(
        c in df.columns
        for c in ("patient_id", "date", "entry_id", "cv_fold",
                  "migraine_target", "migraine_today")
    )
    n_features = df.shape[1] - structural_in_df
    print(
        f"no_rolling feature set: {df.shape[0]} rows x {n_features} features "
        f"(same-day flags, raw exercise minutes, dow)."
    )
    return df
