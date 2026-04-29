import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_LENGTH = 7  # 7-day lookback window
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 0.001
EPOCHS = 50
BATCH_SIZE = 32


class MigraineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MigraineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # We only take the output from the last time step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


def main():
    # Load data
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train_engineered.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val_engineered.parquet"))

    # Features to use (drop identifiers and target)
    cols_to_drop = ['entry_id', 'patient_id', 'date', 'migraine_target']
    X_train_raw = train_df.drop(columns=cols_to_drop).values
    y_train_raw = train_df['migraine_target'].values
    X_val_raw = val_df.drop(columns=cols_to_drop).values
    y_val_raw = val_df['migraine_target'].values

    # Scaling is crucial for LSTMs
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    # Prepare sequences per patient group (simplified for benchmark)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_raw, SEQ_LENGTH)

    # Convert to Tensors
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq)),
                              batch_size=BATCH_SIZE, shuffle=True)

    model = MigraineLSTM(input_size=X_train_raw.shape[1], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting LSTM training...")
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")

    # Save artifacts
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(EXPERIMENT_DIR, "lstm_weights.pt"))
    joblib.dump(scaler, os.path.join(EXPERIMENT_DIR, "scaler.joblib"))
    print("Model and Scaler saved.")


if __name__ == "__main__":
    main()