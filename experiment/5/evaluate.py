import os
import joblib
import pandas as pd
import uuid
from datetime import datetime
from sklearn.metrics import hamming_loss, accuracy_score, f1_score

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FEAT = os.path.join(DATA_DIR, "test_engineered.parquet")
TEST_DISB = os.path.join(DATA_DIR, "test_disability.parquet")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


def main():
    # Load model and artifacts
    artifacts = joblib.load(os.path.join(EXPERIMENT_DIR, "stage5_disability_model.joblib"))
    model = artifacts['model']
    feature_names = artifacts['features']

    # Load test data
    X_test_df = pd.read_parquet(TEST_FEAT)
    y_test_disb = pd.read_parquet(TEST_DISB)

    # Align and test
    test_joined = pd.merge(X_test_df, y_test_disb, on=['patient_id', 'date'], how='inner')

    X_test = test_joined[feature_names]
    y_test = test_joined[['disability_work_affected', 'disability_housework_affected', 'disability_social']].fillna(0)

    # Inference
    y_pred = model.predict(X_test)

    # Metrics for multi-label classification
    h_loss = hamming_loss(y_test, y_pred)  # Fraction of wrong labels
    subset_acc = accuracy_score(y_test, y_pred)  # Exact match ratio

    # Prepare results text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_text = (
        f"STAGE 5: PER-EVENT DISABILITY PREDICTION (EXPLORATORY)\n"
        f"Alignment Count: {len(test_joined)} events\n"
        f"{'-' * 40}\n"
        f"Hamming Loss (lower is better): {h_loss:.4f}\n"
        f"Subset Accuracy (Exact Match):  {subset_acc:.4f}\n"
        f"F1 Score (Macro):               {f1_score(y_test, y_pred, average='macro'):.4f}\n"
    )

    print(output_text)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"results_{timestamp}_{uuid.uuid4()}.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    main()