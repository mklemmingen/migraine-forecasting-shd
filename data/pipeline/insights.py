"""
insights.py - Diagnostic profiling of all pipeline datasets.
"""
import pandas as pd


def print_data_insights(raw_df, translated_df,
                        train_diary, test_diary,
                        train_disability, test_disability,
                        val_diary=None, val_disability=None,
                        label: str = ""):
    """Generate a rigorous scientific data profile for all pipeline datasets.

    val_diary / val_disability are optional - omit for 2-way (train/test) splits.
    label is printed as a header so output from multiple ratios is distinguishable.
    """
    if label:
        print(f"\n{'#' * 120}")
        print(f"# DIAGNOSTICS - SPLIT RATIO: {label}")
        print(f"{'#' * 120}")

    datasets = {
        "1. RAW DIARY DATA": raw_df,
        "2. TRANSLATED DATA": translated_df,
        "3A. DIARY DATA (TRAIN)": train_diary,
    }
    if val_diary is not None:
        datasets["3B. DIARY DATA (VAL)"] = val_diary
    datasets["3C. DIARY DATA (TEST)"] = test_diary
    datasets["4A. DISABILITY SUPPLEMENT (TRAIN)"] = train_disability
    if val_disability is not None:
        datasets["4B. DISABILITY SUPPLEMENT (VAL)"] = val_disability
    datasets["4C. DISABILITY SUPPLEMENT (TEST)"] = test_disability

    for name, df in datasets.items():
        total_rows = len(df)
        total_cols = len(df.columns)

        print("=" * 120)
        print(f" {name}")
        print(f" SHAPE: {total_rows:,} rows | {total_cols:,} columns")

        if total_rows == 0:
            print("-" * 120)
            print(" [EMPTY DATAFRAME - check split logic]")
            print("\n")
            continue

        if 'patient_id' in df.columns and 'date' in df.columns:
            n_patients = df['patient_id'].nunique()
            date_min, date_max = df['date'].min(), df['date'].max()
            days_per_patient = df.groupby('patient_id').size()
            dupes = df.duplicated(subset=['patient_id', 'date']).sum()

            print("-" * 120)
            print(f" COHORT INTEGRITY:")
            if pd.notna(date_min) and pd.notna(date_max):
                print(f" • Patients: {n_patients} | Date Range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
            print(f" • Days per patient -> Min: {days_per_patient.min()} | Median: {days_per_patient.median():.1f} | Max: {days_per_patient.max()}")
            print(f" • Duplicate (Patient, Date) rows: {dupes} " + ("(WARNING!)" if dupes > 0 else "(Clean)"))

        print("-" * 120)

        for col in df.columns:
            if isinstance(col, tuple):
                col_name = " - ".join([str(c) for c in col if pd.notna(c) and not str(c).startswith('Unnamed:')])
            else:
                col_name = str(col)

            dtype = str(df[col].dtype)
            null_count = df[col].isna().sum()
            null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
            unique_count = df[col].nunique()

            info_str = f"• {col_name[:35]:<35} | {dtype:<10} | Nulls: {null_count:>5} ({null_pct:>5.1f}%) | Unq: {unique_count:>4}"

            if unique_count > 0 and unique_count <= 5 and not pd.api.types.is_float_dtype(df[col]):
                val_counts = df[col].value_counts(dropna=False).to_dict()
                dist_str = ", ".join([f"{k}: {v}" for k, v in val_counts.items()])
                info_str += f" | Dist: [{dist_str}]"
            elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                std_val = df[col].std()
                if pd.notna(min_val):
                    info_str += f" | Min: {min_val:>6.1f} | Max: {max_val:>6.1f} | Mean: {mean_val:>6.1f} | Std: {std_val:>6.1f}"

            print(info_str)

        print("\n")
