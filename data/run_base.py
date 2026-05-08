import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (translate_sheet3, engineer_features, assign_cv_folds,
                      process_disability_sheet)

DATA_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_XLS       = os.path.join(DATA_DIR, "raw", "SHD-Dataset.xls")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

raw_df = None
translated_df = None
engineered_df = None
disability_df = None

def runBase():
    global raw_df, translated_df, engineered_df, disability_df
    # Step 1 — Translation
    raw_df = pd.read_excel(RAW_XLS, sheet_name=2, header=[1, 2])
    translated_df = translate_sheet3(raw_df)
    translated_df.to_parquet(os.path.join(PROCESSED_DIR, "translated.parquet"), index=False)
    print("Saved: translated.parquet")

    # Step 2 — Engineering (no split inside)
    engineered_df = engineer_features(translated_df)
    (engineered_df
     .drop(columns=['migraine_today'], errors='ignore')
     .to_parquet(os.path.join(PROCESSED_DIR, "diary.parquet"), index=False))
    print(f"Saved: diary.parquet  ({len(engineered_df)} rows)")

    # CV — split-agnostic, idempotent across run scripts
    cv_df = assign_cv_folds(engineered_df.drop(columns=['migraine_today'], errors='ignore'))
    cv_df.to_parquet(os.path.join(PROCESSED_DIR, "diary_cv5_timeseries.parquet"), index=False)
    print(
        f"Saved: diary_cv5_timeseries.parquet  ({len(cv_df)} rows)  folds: {dict(cv_df['cv_fold'].value_counts().sort_index())}")

    # Step 3 — Disability (full, unsplit)
    raw_dis_df = pd.read_excel(RAW_XLS, sheet_name=1, header=[1, 2])
    disability_df = process_disability_sheet(raw_dis_df)
    disability_df.to_parquet(os.path.join(PROCESSED_DIR, "disability.parquet"), index=False)
    print(f"Saved: disability.parquet  ({len(disability_df)} rows)")

def main():
    runBase()

if __name__ == "__main__":
    main()