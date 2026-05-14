"""filter_to_park_features.py - Park et al. (2016) [1, Tab. 4, p. 8]
stepwise-selected trigger set.

Park et al. ran a stepwise multiple logistic regression over 153
possible combinations of the 18 candidate triggers in their SHD
dataset to identify which features significantly discriminate migraine
headaches from non-migraine headaches. Six individual triggers were
selected by that procedure:

  - Stress              (OR 1.8, 95% CI 1.4-2.4, p<0.001)
  - Hormonal changes    (OR 3.5, 95% CI 2.3-5.2, p<0.001)
  - Noise               (OR 2.8, 95% CI 1.4-4.9, p=0.002)
  - Alcohol             (OR 2.5, 95% CI 1.3-5.0, p=0.009)
  - Overeating          (OR 2.4, 95% CI 1.1-5.7, p=0.009)
  - Traveling           (OR 6.4, 95% CI 1.2-10.2, p=0.003)

The other 12 triggers analysed (excessive sleep, exercise, fatigue,
emotional changes, weather changes, sunlight, odors, fasting,
caffeine, smoking, cheese/chocolate, sleep deprivation) were marked
"NA, not available due lack of inclusion in the stepwise multiple
regression analysis" in Park et al. Table 4 [1, Tab. 4, p. 8]. The
filter here keeps only the 6 stepwise-selected triggers.

Hormonal-changes representation: Park's analysis used "hormonal
changes" as a single trigger from the baseline 18-trigger inventory
[1, Methods, p. 3]. The Korean SHD source records this as two distinct
columns (월경기 = menstruation, 배란기 = ovulation). To match Park's
analytic representation, this filter combines them as
``hormonal_changes_today = menstruation_today OR ovulation_today``.

Preventive medication: Park et al. uses ``preventive_medication`` as
a stratifier in Table 5 [1, Tab. 5, p. 9] - not as a trigger in
Table 4's stepwise model. The current engineered parquets in this
benchmark do not carry the ``preventive_medication`` column (a
pre-existing translation-stage gap, unrelated to this feature set's
definition), so it is excluded here as a data-availability matter.

Same-day vs next-day: Park's analysis was for same-day migraine vs
non-migraine headache discrimination. This benchmark predicts
next-day migraine state. Using today's Park-selected triggers as
features for tomorrow's migraine is a defensible scientific choice
under the carry-over hypothesis but is NOT exactly what Park studied.
See docs/park_features.md for the full framing.

Park's stepwise model also included two interaction terms
(``stress × hormonal_changes``, ``noise × travel``); these are not
materialised as explicit features here, since gradient-boosted trees
and TabPFN can both learn pairwise interactions from the constituent
features themselves.
"""
from pathlib import Path

import pandas as pd

from _dataRead._select_columns import select_columns


# Columns that must be present in the engineered parquet for this
# feature set to be well-defined. Two of them (menstruation_today,
# ovulation_today) are combined into a single derived feature inside
# this loader before being returned.
_PARK_REQUIRED_COLUMNS = (
    "patient_id",
    "date",
    "migraine_target",
    "stress_today",
    "menstruation_today",
    "ovulation_today",
    "noise_today",
    "alcohol_today",
    "overeating_today",
    "travel_today",
)

# Identifier columns that pass through if present but are tolerated if
# absent. ``cv_fold`` only exists in CV-split parquets; ``entry_id``
# may or may not be present depending on the upstream pipeline run.
_PARK_OPTIONAL_COLUMNS = (
    "entry_id",
    "cv_fold",
)


def select_park_features(parquet_path: str | Path) -> pd.DataFrame:
    """Return a DataFrame containing only the Park-stepwise feature set.

    Output columns:
      Structural: ``patient_id``, ``date``, ``migraine_target``
                  (+ ``entry_id``, ``cv_fold`` if present)
      Features (6, as Park's stepwise selected):
        - ``stress_today``
        - ``hormonal_changes_today`` (derived: menstruation OR ovulation)
        - ``noise_today``
        - ``alcohol_today``
        - ``overeating_today``
        - ``travel_today``
    """
    df = select_columns(
        parquet_path,
        required=_PARK_REQUIRED_COLUMNS,
        optional=_PARK_OPTIONAL_COLUMNS,
    )

    # Combine the two Korean hormonal sub-fields into Park's single
    # "hormonal changes" trigger. Both are 0/1 flags in the engineered
    # parquet; logical OR preserves "any hormonal-cycle day".
    df["hormonal_changes_today"] = (
        df["menstruation_today"].astype(int) | df["ovulation_today"].astype(int)
    ).astype(int)
    df = df.drop(columns=["menstruation_today", "ovulation_today"])

    structural_in_df = sum(
        c in df.columns
        for c in ("patient_id", "date", "entry_id", "cv_fold", "migraine_target")
    )
    n_features = df.shape[1] - structural_in_df
    print(
        f"Park feature set: {df.shape[0]} rows x {n_features} features "
        f"(Park et al. 2016 Tab. 4 stepwise-selected: stress, "
        f"hormonal_changes [menstruation OR ovulation], noise, alcohol, "
        f"overeating, travel)."
    )
    return df
