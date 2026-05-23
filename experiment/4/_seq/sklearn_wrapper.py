"""SequenceClassifier - sklearn-style wrapper that windows internally.

Honours the Addition builder contract (.predict_proba(X) -> (n, 2),
row-aligned) while consuming an id-bearing X so it can build per-patient
look-back windows. This is what keeps experiment/4's evaluate.py identical to
Additions 0/1: the leaf still calls ``model.predict_proba(X)[:, 1]`` and the
probabilities line up 1:1 with the rows the bootstrap evaluator scores.

The actual network is supplied as a ``module_factory`` so the three
architectures (GRU, TCN, window-MLP) share one fit/predict/serialisation
path and differ only in their nn.Module. Each build_<arch> in
_model_architecture/<arch>/model.py instantiates this class with its factory.

Imbalance is handled by a weighted training objective (not resampling) so the
output probabilities stay interpretable for the calibration panel
(Brier/ECE10/slope) the benchmark reports - the same policy as the tree
leaves, which leave imbalance to the external threshold step.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from windowing import make_windows, feature_columns, DATE_COL  # addition-local (sys.path)

# torch is the existing deep-learning dependency (ROCm stack, docs/tabPfn.MD).
import torch
import torch.nn as nn

# module_factory(n_features, lookback, hidden, dropout) -> nn.Module
#   forward(sequences, mask) -> logits of shape (batch,)   (single positive-class logit)
ModuleFactory = Callable[[int, int, int, float], nn.Module]


class SequenceClassifier:
    """Self-windowing binary sequence classifier, sklearn-compatible."""

    def __init__(
        self,
        module_factory: ModuleFactory,
        *,
        lookback: int = 7,
        gap_segment_days: int = 7,
        hidden: int = 16,
        dropout: float = 0.3,
        weight_decay: float = 1e-3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 128,
        device: str = "cuda",
        random_state: int = 0,
    ):
        self.module_factory = module_factory
        self.lookback = lookback
        self.gap_segment_days = gap_segment_days
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.random_state = random_state
        # Set at fit time:
        self.classes_ = np.array([0, 1])
        self.feature_cols_: Optional[List[str]] = None
        self.module_: Optional[nn.Module] = None

    # -- public sklearn surface ------------------------------------------------

    def fit(self, X: pd.DataFrame, y) -> "SequenceClassifier":
        """Lock the feature columns, window X, and train the network."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.feature_cols_ = feature_columns(X)

        batch = make_windows(
            X, lookback=self.lookback, gap_segment_days=self.gap_segment_days,
            feature_cols=self.feature_cols_,
        )
        y_arr = np.asarray(y, dtype=np.float32)[batch.row_index]
        dates = pd.to_datetime(X[DATE_COL]).to_numpy()[batch.row_index]
        self.module_ = self.module_factory(
            batch.n_features, self.lookback, self.hidden, self.dropout
        ).to(self.device)

        self._train(batch.sequences, batch.mask, y_arr, dates, self._pos_weight(y_arr))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Row-aligned class probabilities, shape (len(X), 2)."""
        if self.module_ is None or self.feature_cols_ is None:
            raise RuntimeError("predict_proba called before fit")
        batch = make_windows(
            X, lookback=self.lookback, gap_segment_days=self.gap_segment_days,
            feature_cols=self.feature_cols_,
        )
        p1 = self._forward_probs(batch.sequences, batch.mask)  # (n_windows,)
        out = np.zeros((len(X), 2), dtype=float)
        out[batch.row_index, 1] = p1
        out[batch.row_index, 0] = 1.0 - p1
        return out

    # -- imbalance objective (Decision 3) --------------------------------------

    def _pos_weight(self, y_arr: np.ndarray) -> float:
        """Positive-class weight for the BCE loss.

        Decision 3 (docs/addition4_sequence.md Section 9): NO imbalance
        correction. Reweighting/resampling does not improve discrimination and
        strongly miscalibrates the minority-class probability
        [vandengoorbergh2022imbalance, p. 1525; p. 1530], and this benchmark is
        calibration-first. Imbalance is handled downstream by the external
        operating-threshold step, matching the TabPFN (Addition 1) policy. So
        the weight is 1.0 (unweighted BCE). A clamped weight is the documented
        sensitivity fallback if a cell degenerates, never the default.
        """
        del y_arr
        return 1.0

    # -- training and forward (Decision 4) -------------------------------------

    def _train(self, sequences, mask, y_arr, dates, pos_weight) -> None:
        """Train ``self.module_`` with unweighted BCE + chronological early stop.

        Small capacity and non-trivial weight decay keep overfitting in check:
        ~201 positive migraine days at EPV 3.9 (docs/dataset.md;
        martin2025samplesize p. 2; grinsztajn2022tabular p. 1). Early stopping
        uses a chronological tail of the TRAIN windows only - never val/test.
        """
        dev = self.device
        Xt = torch.tensor(sequences, dtype=torch.float32, device=dev)
        Mt = torch.tensor(mask, dtype=torch.float32, device=dev)
        yt = torch.tensor(y_arr, dtype=torch.float32, device=dev)

        n = len(y_arr)
        order = np.argsort(dates, kind="stable")
        cut = int(n * 0.85)
        if 0 < cut < n:
            tr_idx, es_idx = order[:cut], order[cut:]
        else:
            tr_idx, es_idx = order, np.array([], dtype=int)
        use_es = len(es_idx) > 0 and len(np.unique(y_arr[es_idx])) > 1
        tr = torch.tensor(tr_idx, device=dev)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float(pos_weight), device=dev))
        opt = torch.optim.AdamW(
            self.module_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state, best_loss, patience, bad = None, float("inf"), 10, 0
        for _epoch in range(self.epochs):
            self.module_.train()
            perm = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = criterion(self.module_(Xt[b], Mt[b]), yt[b])
                loss.backward()
                opt.step()
            if use_es:
                self.module_.eval()
                es = torch.tensor(es_idx, device=dev)
                with torch.no_grad():
                    es_loss = criterion(self.module_(Xt[es], Mt[es]), yt[es]).item()
                if es_loss < best_loss - 1e-4:
                    best_loss, bad = es_loss, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self.module_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.module_.load_state_dict(best_state)

    def _forward_probs(self, sequences, mask) -> np.ndarray:
        """Run the fitted module in eval mode; return positive-class probs,
        aligned to the make_windows row order (i.e. batch.row_index)."""
        dev = self.device
        self.module_.eval()
        Xt = torch.tensor(sequences, dtype=torch.float32, device=dev)
        Mt = torch.tensor(mask, dtype=torch.float32, device=dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), self.batch_size):
                logits = self.module_(Xt[i:i + self.batch_size], Mt[i:i + self.batch_size])
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=float)

    # -- joblib serialisation note ---------------------------------------------
    # train.py does joblib.dump(model). A CUDA-resident nn.Module pickles but
    # reloads onto the same device; move the module to CPU before returning
    # from build_<arch> (or implement __getstate__/__setstate__ here) so the
    # pickle is portable. Left as a deliberate decision point for the impl.
