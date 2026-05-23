"""Window-MLP sequence classifier (Addition 4).

Flattens the last N days of features and feeds a small dense network. It is a
"sequence model" only in that it sees N days at once, which makes it the
cleanest test of RQ2 in docs/addition4_sequence.md Sec. 2: does seeing the
RAW look-back window beat seeing the ENGINEERED lag/rolling summary of that
window that the tabular models in Additions 0/1 already receive? If the
window-MLP matches the tabular full_features model while consuming only
same-day features (no_rolling_features), the engineered lags add nothing the
window does not already contain.
"""
import sys
from pathlib import Path

import torch.nn as nn

_ADDITION_ROOT = Path(__file__).resolve().parents[2]
if str(_ADDITION_ROOT / "_seq") not in sys.path:
    sys.path.insert(0, str(_ADDITION_ROOT / "_seq"))
from sklearn_wrapper import SequenceClassifier  # noqa: E402


class _WindowMLPNet(nn.Module):
    def __init__(self, n_features: int, lookback: int, hidden: int, dropout: float):
        super().__init__()
        in_dim = n_features * lookback  # flattened window (+ optionally the mask)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, sequences, mask):
        # sequences: (B, lookback, n_features); mask: (B, lookback)
        x = sequences * mask.unsqueeze(-1)        # zero padded (left-pad) steps
        x = x.reshape(x.shape[0], -1)             # flatten window -> (B, in_dim)
        return self.net(x).squeeze(-1)            # (B,) positive-class logits


def build_window_mlp(X_train, y_train, *, device="cuda", random_state=0, output_dir=None):
    """Fit the window-MLP SequenceClassifier and return it (CPU-resident)."""
    del output_dir
    model = SequenceClassifier(
        _WindowMLPNet,
        lookback=7,
        gap_segment_days=7,
        hidden=32,
        dropout=0.4,
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
