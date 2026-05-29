import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (translate_sheet3, engineer_features, assign_cv_folds,
                      process_disability_sheet)
from pipeline._special import extract_aura_status

DATA_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_XLS       = os.path.join(DATA_DIR, "raw", "SHD-Dataset.xls")
BASE_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPECIAL_DIR   = os.path.join(BASE_PROCESSED_DIR, "special")
PROCESSED_DIR = None

raw_df = None
translated_df = None
engineered_df = None
disability_df = None

def runBase(target_mode: str = "headache"):
    global raw_df, translated_df, engineered_df, disability_df

    PROCESSED_DIR = os.path.join(BASE_PROCESSED_DIR, target_mode)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1 - Translation
    raw_df = pd.read_excel(RAW_XLS, sheet_name=2, header=[1, 2])
    translated_df = translate_sheet3(raw_df)
    translated_df.to_parquet(os.path.join(PROCESSED_DIR, "translated.parquet"), index=False)
    print("Saved: translated.parquet")

    # Step 2 - Engineering (no split inside)
    engineered_df = engineer_features(
        translated_df,
        target_mode=target_mode,
        disability_df=disability_df if target_mode == "migraine" else None,
    )
    (engineered_df
     .drop(columns=['migraine_today'], errors='ignore')
     .to_parquet(os.path.join(PROCESSED_DIR, "diary.parquet"), index=False))
    print(f"Saved: diary.parquet  ({len(engineered_df)} rows)")

    # CV - split-agnostic, idempotent across run scripts
    cv_df = assign_cv_folds(engineered_df.drop(columns=['migraine_today'], errors='ignore'))
    cv_df.to_parquet(os.path.join(PROCESSED_DIR, "diary_cv5_timeseries.parquet"), index=False)
    print(
        f"Saved: diary_cv5_timeseries.parquet  ({len(cv_df)} rows)  folds: {dict(cv_df['cv_fold'].value_counts().sort_index())}")

    # Step 3 - Disability (full, unsplit)
    raw_dis_df = pd.read_excel(RAW_XLS, sheet_name=1, header=[1, 2])
    disability_df = process_disability_sheet(raw_dis_df)
    disability_df.to_parquet(os.path.join(PROCESSED_DIR, "disability.parquet"), index=False)
    print(f"Saved: disability.parquet  ({len(disability_df)} rows)")

    # Step 4 (special) - Per-patient phenotype flags that are absent from the
    # engineered diary features but needed for sensitivity analyses. Persisted
    # once to data/processed/special/ rather than per target-mode because the
    # flags are patient-level baseline characteristics shared by both targets.
    os.makedirs(SPECIAL_DIR, exist_ok=True)
    aura_df = extract_aura_status(RAW_XLS)
    aura_df.to_parquet(os.path.join(SPECIAL_DIR, "aura_status.parquet"), index=False)
    n_aura = int(aura_df["has_aura"].sum())
    print(f"Saved: special/aura_status.parquet  ({len(aura_df)} patients, {n_aura} with aura)")

def main():
    runBase()

if __name__ == "__main__":
    main()