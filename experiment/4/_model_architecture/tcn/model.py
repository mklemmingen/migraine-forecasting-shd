"""Temporal convolutional network sequence classifier (Addition 4).

A small 1-D causal-convolution stack over the look-back window, receptive
field capped at the window length. The convolutional alternative of
docs/addition4_sequence.md Sec. 3.1 - often cheaper and more stable than
recurrence on short sequences, and a useful second architecture for the
falsification test (does any sequence model beat the engineered lags?).
"""
import sys
from pathlib import Path

import torch.nn as nn

_ADDITION_ROOT = Path(__file__).resolve().parents[2]
if str(_ADDITION_ROOT / "_seq") not in sys.path:
    sys.path.insert(0, str(_ADDITION_ROOT / "_seq"))
from sklearn_wrapper import SequenceClassifier  # noqa: E402


class _TCNNet(nn.Module):
    def __init__(self, n_features: int, lookback: int, hidden: int, dropout: float):
        super().__init__()
        # Two shallow conv blocks; kernel <= lookback, same-padding to preserve
        # length so the last timestep stays aligned to the current day. Shallow
        # by design given the EPV-3.9 overfitting constraint.
        k = 3 if lookback >= 3 else max(1, lookback)
        pad = k // 2
        self.conv1 = nn.Conv1d(n_features, hidden, kernel_size=k, padding=pad)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=k, padding=pad)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, sequences, mask):
        # (B, lookback, n_features) -> Conv1d wants (B, C=n_features, L)
        x = (sequences * mask.unsqueeze(-1)).transpose(1, 2)
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        h = x[:, :, -1]                           # last timestep = current day
        return self.head(h).squeeze(-1)           # (B,) logits


def build_tcn(X_train, y_train, *, device="cuda", random_state=0, output_dir=None):
    """Fit the TCN SequenceClassifier and return it (CPU-resident for joblib)."""
    del output_dir
    model = SequenceClassifier(
        _TCNNet,
        lookback=7,
        gap_segment_days=7,
        hidden=16,
        dropout=0.3,
        weight_decay=1e-3,
        lr=1e-3,
        epochs=100,
        device=device,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    if model.module_ is not None:
        model.module_.to("cpu")
        model.device = "cpu"
    return model
