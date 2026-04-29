import os
import torch
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef, recall_score
from collections import defaultdict

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(EXPERIMENT_DIR, "final_model")
TEST_PATH = os.path.join(DATA_DIR, "test_engineered.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val_engineered.parquet")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


# Reuse serialization logic from train.py
def serialize_row(row):
    triggers = [col.replace('_today', '') for col in row.index if '_today' in col and row[col] == 1]
    trigger_str = ", ".join(triggers) if triggers else "no specific triggers"
    return (f"Patient report for today: The state is {'migraine' if row['migraine_today'] == 1 else 'headache-free'}. "
            f"Triggers endorsed: {trigger_str}. Risk tomorrow.")


def get_predictions(model, tokenizer, texts):
    model.eval()
    probs = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            probs.append(softmax(outputs.logits.numpy(), axis=1)[0][1])
    return np.array(probs)


def main():
    print("Loading fine-tuned model and test data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    val_texts = val_df.apply(serialize_row, axis=1).tolist()
    test_texts = test_df.apply(serialize_row, axis=1).tolist()

    y_val = val_df['migraine_target'].values
    y_test = test_df['migraine_target'].values

    print("Inference on Validation and Test sets...")
    y_prob_val = get_predictions(model, tokenizer, val_texts)
    y_prob_test = get_predictions(model, tokenizer, test_texts)

    # Thresholding and Bootstrapping (Omitted here for brevity, reuse logic from Stage 0)
    # ... [Insert find_operating_thresholds and run_bootstrap_evaluation from Stage 0] ...

    # Output to File
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"results_{timestamp}_{uuid.uuid4()}.txt"

    with open(os.path.join(RESULTS_DIR, filename), "w") as f:
        f.write(f"STAGE 3: MEDICALLY PRETRAINED TRANSFORMER RESULTS\n")
        f.write(f"Base Model: {MODEL_PATH}\n")
        # Write metrics...

    print(f"Evaluation complete. Results saved to {filename}")


if __name__ == "__main__":
    main()