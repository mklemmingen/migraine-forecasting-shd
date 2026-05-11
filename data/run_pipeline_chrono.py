"""
run_pipeline_chrono.py - Full pipeline with chronological date-percentile split.

Outputs written to data/processed/:
  translated.parquet              (full, pre-engineering)
  diary.parquet                   (full engineered, no split)
  disability.parquet              (full disability, no split)
  diary_cv5_timeseries.parquet    (full engineered with 5-fold time-series CV labels)
  70_15_15/chrono/
    diary_train.parquet  diary_val.parquet  diary_test.parquet
    disability_train.parquet  disability_val.parquet  disability_test.parquet
  80_20/chrono/
    diary_train.parquet  diary_test.parquet
    disability_train.parquet  disability_test.parquet
  70_30/chrono/
    diary_train.parquet  diary_test.parquet
    disability_train.parquet  disability_test.parquet
"""
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import run_base

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from pipeline import print_data_insights
from pipeline.analytics import run_dataset_analysis, run_class_balance
from pipeline.splits.chrono import apply_split, SPLIT_NAME

DATA_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_XLS       = os.path.join(DATA_DIR, "raw", "SHD-Dataset.xls")
BASE_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PROCESSED_DIR = None

SPLIT_RATIOS = {
    "70_15_15": {"cutpoints": (0.70, 0.85), "folds": ("train", "val", "test")},
    "80_20":    {"cutpoints": (0.80,),       "folds": ("train", "test")},
    "70_30":    {"cutpoints": (0.70,),       "folds": ("train", "test")},
}


def main():

    for target_mode in ("headache", "migraine"):

        PROCESSED_DIR = os.path.join(BASE_PROCESSED_DIR, target_mode)

        print(f"\n=== Processing target_mode: {target_mode} ===\n")

        run_base.runBase(target_mode)

        # Step 4 - Generate splits for every ratio
        for ratio_name, cfg in SPLIT_RATIOS.items():
            split_dir = os.path.join(PROCESSED_DIR, ratio_name, SPLIT_NAME)
            os.makedirs(split_dir, exist_ok=True)

            diary_split, boundaries = apply_split(
                run_base.engineered_df,
                cutpoints=cfg["cutpoints"],
                fold_names=cfg["folds"],
            )
            for fold in cfg["folds"]:
                subset = (diary_split[diary_split['split'] == fold]
                          .drop(columns=['split', 'migraine_today'], errors='ignore'))
                subset.to_parquet(os.path.join(split_dir, f"diary_{fold}.parquet"), index=False)
                print(f"Saved: {ratio_name}/{SPLIT_NAME}/diary_{fold}.parquet  ({len(subset)} rows)")

            dis_split, _ = apply_split(run_base.disability_df, boundaries=boundaries)
            for fold in cfg["folds"]:
                subset = (dis_split[dis_split['split'] == fold]
                          .drop(columns=['split'], errors='ignore'))
                subset.to_parquet(os.path.join(split_dir, f"disability_{fold}.parquet"), index=False)
                print(f"Saved: {ratio_name}/{SPLIT_NAME}/disability_{fold}.parquet  ({len(subset)} rows)")

            has_val = 'val' in cfg["folds"]
            print_data_insights(
                run_base.raw_df, run_base.translated_df,
                diary_split[diary_split['split'] == 'train'].drop(columns=['split', 'migraine_today'], errors='ignore'),
                diary_split[diary_split['split'] == 'test'].drop(columns=['split', 'migraine_today'], errors='ignore'),
                dis_split[dis_split['split'] == 'train'].drop(columns=['split'], errors='ignore'),
                dis_split[dis_split['split'] == 'test'].drop(columns=['split'], errors='ignore'),
                val_diary=diary_split[diary_split['split'] == 'val'].drop(columns=['split', 'migraine_today'], errors='ignore') if has_val else None,
                val_disability=dis_split[dis_split['split'] == 'val'].drop(columns=['split'], errors='ignore') if has_val else None,
                label=ratio_name,
            )

        print(f"\nPipeline complete ({SPLIT_NAME}). Parquets in data/processed/\n")

        # Step 5 - Analytics reports
        PROCESSED = Path(DATA_DIR) / "processed"
        tm_dir = PROCESSED / target_mode
        diary_df = pd.read_parquet(tm_dir / "diary.parquet")
        run_dataset_analysis(diary_df=diary_df, target_mode=target_mode,
                             output_path=tm_dir / "dataset_analysis.pdf")
        run_class_balance(processed_dir=tm_dir, target_mode=target_mode,
                          output_path=tm_dir / "class_balance.pdf")

        print(f"\n=== Processing finished for target_mode: {target_mode} ===\n")


if __name__ == "__main__":
    main()
