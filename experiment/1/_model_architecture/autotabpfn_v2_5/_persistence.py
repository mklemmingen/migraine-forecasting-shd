"""
Persistence handle for AutoTabPFN leaves.

Pickling a fitted ``AutoTabPFNClassifier`` directly causes a segmentation
fault under PyTorch 2.11.0+rocm7.2 (HIP 7.2.26015 on AMD): joblib walks
the wrapper's torch-tensor references during ``getstate``, and the AMD
runtime crashes inside the destructor cascade. The crash happens after
AutoGluon's own ``TabularPredictor.save()`` succeeds, so the on-disk
predictor is intact - only the joblib step that the leaf's generic
``train.py`` template runs fails.

``AutoTabPFNHandle`` sidesteps the issue by pickling only the
``TabularPredictor``'s on-disk path as a string. On first call to
``predict_proba`` the handle lazily loads the predictor from disk via
``TabularPredictor.load`` and forwards inference to it. The handle's
``__getstate__`` / ``__setstate__`` enforce that the predictor object
itself is never put into the pickle, even if a caller has cached it
on the handle's lazy slot.

The leaf's ``evaluate.py`` template loads the handle with
``joblib.load(MODEL_PATH)`` and calls ``handle.predict_proba(X)[:, 1]``
exactly as it would for any other variant - no per-variant evaluator
branching required.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor


class AutoTabPFNHandle:
    """Picklable lazy-loading wrapper around an AutoGluon ``TabularPredictor``.

    Stores the predictor's on-disk path as a string; the predictor object
    is loaded on first inference call and cached on the instance, but is
    never included in the pickled state. The handle's pickled size is
    dominated by the path string (~200 bytes), so ``joblib.dump`` never
    touches GPU-resident state.
    """

    __slots__ = ("predictor_path", "_predictor")

    def __init__(self, predictor_path: str | Path) -> None:
        self.predictor_path: str = str(predictor_path)
        self._predictor: "TabularPredictor | None" = None

    def _load(self) -> "TabularPredictor":
        if self._predictor is None:
            from autogluon.tabular import TabularPredictor
            self._predictor = TabularPredictor.load(self.predictor_path)
        return self._predictor

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return shape-(n_samples, n_classes) probability array.

        Matches the sklearn ``predict_proba`` contract that the leaf's
        ``evaluate.py`` expects so the downstream
        ``model.predict_proba(X)[:, 1]`` pattern works unchanged.
        AutoGluon's ``TabularPredictor.predict_proba`` returns a
        ``DataFrame`` with one column per class; the column order is
        preserved by AutoGluon and matches the training-time class order.
        """
        predictor = self._load()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        proba = predictor.predict_proba(X)
        return proba.values if hasattr(proba, "values") else np.asarray(proba)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        predictor = self._load()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        preds = predictor.predict(X)
        return preds.values if hasattr(preds, "values") else np.asarray(preds)

    def __getstate__(self) -> dict:
        # Never pickle the predictor object itself - that is the
        # whole point of this handle.
        return {"predictor_path": self.predictor_path}

    def __setstate__(self, state: dict) -> None:
        self.predictor_path = state["predictor_path"]
        self._predictor = None

    def __repr__(self) -> str:
        loaded = "loaded" if self._predictor is not None else "lazy"
        return f"AutoTabPFNHandle(predictor_path={self.predictor_path!r}, state={loaded})"
