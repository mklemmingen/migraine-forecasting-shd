import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FEAT = os.path.join(DATA_DIR, "train_engineered.parquet")
TRAIN_DISB = os.path.join(DATA_DIR, "train_disability.parquet")
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "stage5_disability_model.joblib")


def main():
    print("Loading and joining features with disability outcomes...")
    # Load daily triggers (X) and disability events (y)
    X_df = pd.read_parquet(TRAIN_FEAT)
    y_df = pd.read_parquet(TRAIN_DISB)

    # Join on patient_id and date to ensure temporal alignment
    # Note: Only days WITH a headache have disability labels
    joined = pd.merge(X_df, y_df, on=['patient_id', 'date'], how='inner')

    # Define targets: Work, Housework, and Social disability
    target_cols = [
        'disability_work_affected',
        'disability_housework_affected',
        'disability_social'
    ]

    # Define features (Triggers + Migraine History)
    drop_cols = ['entry_id_x', 'entry_id_y', 'patient_id', 'date', 'migraine_target'] + target_cols
    # Also drop Sheet 2 specific clinical features to avoid leakage (e.g., severity_vas)
    clinical_leakage = ['severity_category', 'severity_vas', 'migraine_flag', 'disability_any']

    X = joined.drop(columns=drop_cols + clinical_leakage, errors='ignore')
    y = joined[target_cols].fillna(0).astype(int)

    print(f"Training on {len(joined)} identified headache events...")

    # Multi-Output RF: Captures correlations between different disability domains
    forest = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)

    multi_target_forest.fit(X, y)

    # Save model and feature names
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    joblib.dump({'model': multi_target_forest, 'features': X.columns.tolist()}, MODEL_PATH)
    print(f"Exploratory disability model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()