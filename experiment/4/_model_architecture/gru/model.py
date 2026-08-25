"""GRU sequence classifier (Addition 4).

A single small gated-recurrent layer over the look-back window, with the
last (most recent) timestep's hidden state mapped to one positive-class
logit. The default recurrent baseline of docs/addition4_sequence.md Sec. 3.1;
kept deliberately small (8-32 units) because the migraine cell sits at
events-per-variable 3.9 (docs/dataset.md), the high-risk overfitting band.
"""
import sys
from pathlib import Path

import torch.nn as nn

# _seq/ is at experiment/4/_seq; add the addition root so it imports cleanly
# whether called from train.py (which already adds it) or in isolation.
_ADDITION_ROOT = Path(__file__).resolve().parents[2]
if str(_ADDITION_ROOT / "_seq") not in sys.path:
    sys.path.insert(0, str(_ADDITION_ROOT / "_seq"))
from sklearn_wrapper import SequenceClassifier  # noqa: E402

import torch as _torch

# Prefer a GPU when one is actually usable; fall back to CPU otherwise. The
# previous hardcoded 'cuda' default made every builder raise
# "Torch not compiled with CUDA enabled" on CPU-only machines.
_DEFAULT_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"


class _GRUNet(nn.Module):
    def __init__(self, n_features: int, lookback: int, hidden: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, sequences, mask):
        # sequences: (B, lookback, n_features); mask: (B, lookback).
        # Windows are LEFT-padded and right-aligned, so the last timestep is
        # always the observed current day - its output is the readout (no
        # packing/gather needed).
        x = sequences * mask.unsqueeze(-1)        # zero the left-pad steps
        out, _ = self.gru(x)                      # (B, lookback, hidden)
        h = out[:, -1, :]                         # last step = current day
        return self.head(self.drop(h)).squeeze(-1)  # (B,) logits


def build_gru(X_train, y_train, *, device=_DEFAULT_DEVICE, random_state=0, output_dir=None):
    """Fit the GRU SequenceClassifier and return it (CPU-resident for joblib)."""
    del output_dir
    model = SequenceClassifier(
        _GRUNet,
        lookback=7,          # short window: Addition 3 found dependence decays by day 3-7
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
