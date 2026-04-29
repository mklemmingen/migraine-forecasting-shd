import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score
import uuid

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "nlpie/distilbiobert"  # Efficient medical LLM
TRAIN_PATH = os.path.join(DATA_DIR, "train_engineered.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val_engineered.parquet")
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")


def serialize_row(row):
    """Converts a tabular row into a clinical descriptive sentence."""
    triggers = [col.replace('_today', '') for col in row.index if '_today' in col and row[col] == 1]
    trigger_str = ", ".join(triggers) if triggers else "no specific triggers"

    text = (f"Patient report for today: The state is {'migraine' if row['migraine_today'] == 1 else 'headache-free'}. "
            f"Triggers endorsed: {trigger_str}. "
            f"Sleep debt level: {row['sleep_debt_3day']}. "
            f"Stress level: {'high' if row['stress_today'] == 1 else 'normal'}. "
            f"Forecasting risk for tomorrow.")
    return text


class MigraineDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return {"auroc": roc_auc_score(labels, probs)}


def main():
    print("Loading and serializing data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    train_texts = train_df.apply(serialize_row, axis=1).tolist()
    val_texts = val_df.apply(serialize_row, axis=1).tolist()

    y_train = train_df['migraine_target'].astype(int).tolist()
    y_val = val_df['migraine_target'].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = MigraineDataset(train_encodings, y_train)
    val_dataset = MigraineDataset(val_encodings, y_val)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning Medically Pretrained Transformer...")
    trainer.train()

    # Save the final model and tokenizer
    model.save_pretrained(os.path.join(EXPERIMENT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(EXPERIMENT_DIR, "final_model"))


if __name__ == "__main__":
    main()