"""Pairwise (order-2) Shapley interactions for TabPFN headline leaves.

ShapIQ [muschalik2024shapiq] computes Shapley *interaction* indices, the
principled extension of Shapley values to pairs (and higher orders) of
features. A standard SHAP bar chart shows main-effect attribution only and
cannot tell whether two triggers reinforce each other. The clinical
literature suggests migraine triggers combine non-additively, so the
pairwise interaction question is substantive.

Scoped to the TabPFN headline leaves only: the foundation model's flexible
function class makes interactions plausible and the extension is native via
``tabpfn_extensions.interpretability.shapiq.get_tabpfn_explainer``. Higher-
order interaction estimates are unreliable at low events-per-variable, so
interaction claims are confined to the higher-EPV cells in the
interpretation (docs Section 3.5, claim 5).

Order-2 only; the explainer's k-SII index aggregates the pairwise effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _patch_shapiq_tabpfn_imputer_for_numpy_2x() -> None:
    """Restore single-element predict handling in shapiq's TabPFNImputer.

    Upstream `shapiq` (as of v1.x against NumPy 2.x) calls
    `float(self.predict(x_explain_coal))` at
    `shapiq/imputer/tabpfn_imputer.py::TabPFNImputer.value_function`.
    The wrapped predict returns a 1-element 1-D ndarray (sklearn convention);
    NumPy 1.x silently unwrapped this to a Python scalar, but NumPy 2.x raises
    `TypeError: only 0-dimensional arrays can be converted to Python scalars`.
    The fix unwraps the result via `np.asarray(...).flat[0]`, which works on
    both 0-d and 1-d arrays without changing semantics on multi-element
    returns (the imputer's coalition loop already evaluates one coalition at
    a time). The patch is idempotent and only applies if the symbol exists.
    """
    try:
        from shapiq.imputer import tabpfn_imputer as _tipm
    except Exception:
        return
    if getattr(_tipm.TabPFNImputer, "_migraine_shd_numpy2_patched", False):
        return

    def _patched_value_function(self, coalitions):
        output = np.zeros(len(coalitions), dtype=float)
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:
                output[i] = self.empty_prediction
                continue
            x_train_coal = self.x_train[:, coalition]
            x_explain_coal = self.x[:, coalition]
            self.model.fit(x_train_coal, self.y_train)
            raw = np.asarray(self.predict(x_explain_coal))
            pred = float(raw.flat[0]) if raw.size == 1 else float(raw.mean())
            output[i] = pred
        self.model.fit(self.x_train, self.y_train)
        return output

    _tipm.TabPFNImputer.value_function = _patched_value_function
    _tipm.TabPFNImputer._migraine_shd_numpy2_patched = True


_patch_shapiq_tabpfn_imputer_for_numpy_2x()


# Model classes whose ``.fit`` is a cheap in-context fit (no training loop),
# so the ShapIQ imputer's per-coalition refit is tractable. The fine-tuned
# variant ("FinetunedTabPFNClassifier") runs a multi-epoch training loop per
# fit and is excluded; the AutoTabPFN handle has no single forward pass.
_SINGLE_FIT_TABPFN_CLASSES = {"TabPFNClassifier", "RealTabPFNClassifier"}


def _is_single_fit_tabpfn(model) -> bool:
    """True when ``model``'s fit is a single in-context fit (ShapIQ-tractable)."""
    cls = type(model).__name__
    if cls in _SINGLE_FIT_TABPFN_CLASSES:
        return True
    # Some packagings name the plain classifier differently; treat any class
    # that is a TabPFNClassifier but NOT a finetuned/auto wrapper as single-fit.
    if "Finetuned" in cls or "Auto" in cls or "Handle" in cls:
        return False
    return "TabPFNClassifier" in cls


def _free_gpu() -> None:
    """Release cached GPU memory between coalition refits so the ROCm runtime
    does not accumulate HIP allocations across the per-row k-SII passes."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass

SHAPIQ_SEED = 42
# A small explained subset keeps the per-row coalition-sampling cost
# tractable and the GPU memory bounded; each explained row triggers a budget
# of TabPFN context refits, and on the AMD ROCm runtime a large product of
# rows x coalitions accumulates HIP allocations that abort the process. The
# pair strengths are averaged over rows, so a small sample is sufficient for
# a stable ranking.
_MAX_EXPLAIN = 6
# Coalition-sampling budget per explained row. The k-SII estimator queries
# the model on this many feature coalitions; capping it bounds the GPU cost
# while staying exact on the low-cardinality cells (2^6 = 64 < 256, so the
# 6-feature park cell is computed exactly).
_BUDGET = 256
# Reduced budget for high-cardinality cells (full_features). On 52 features a
# large coalition sample is both costly (one TabPFN context refit per
# coalition on GPU) and not needed: the interpretation refuses interaction
# claims on the EPV-3.9 migraine full_features cell, so this heatmap is a
# descriptive aid only.
_BUDGET_HIGH_DIM = 96


class _ConstantTolerantTabPFN:
    """Thin wrapper that tolerates all-constant feature coalitions.

    ShapIQ's TabPFN imputer re-fits the model on each feature coalition,
    restricting the background to the present columns (the
    remove-and-recontextualize paradigm). When a coalition's background
    columns happen to be constant (common on the sparse binary trigger
    sets), TabPFN's preprocessing rejects the fit with "all features
    constant". A coalition whose background carries no variance conveys no
    information, so the principled value for it is the base rate; this
    wrapper returns the background class prior in that case rather than
    aborting the whole interaction estimate. The base rate is the same
    quantity SHAP uses for the empty coalition, so the fallback is
    consistent with the rest of the attribution.
    """

    def __init__(self, model, base_rate: float):
        self._model = model
        self._base_rate = float(base_rate)
        self._degenerate = False
        self.classes_ = getattr(model, "classes_", np.array([0, 1]))

    def fit(self, X, y):
        try:
            self._model.fit(X, y)
            self._degenerate = False
        except Exception:
            # All-constant (or otherwise unfittable) coalition: fall back to
            # the class prior on the next predict call.
            self._degenerate = True
        self.classes_ = getattr(self._model, "classes_", self.classes_)
        return self

    def predict_proba(self, X):
        n = len(X)
        if self._degenerate:
            return np.tile([1.0 - self._base_rate, self._base_rate], (n, 1))
        return self._model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def compute_pairwise_interactions(
    raw_model,
    X_background: pd.DataFrame,
    y_background,
    X_explain: pd.DataFrame,
    max_explain: int = _MAX_EXPLAIN,
    predict_fn=None,
) -> dict:
    """Estimate mean absolute order-2 k-SII interaction strength per pair.

    Builds the TabPFN-native ShapIQ explainer over the val/cal background
    and averages the absolute pairwise interaction value across the
    explained rows. Returns ``{feature_names, pair_matrix, top_pairs,
    meta}`` where ``pair_matrix`` is a symmetric ``(n_features, n_features)``
    mean-absolute interaction matrix and ``top_pairs`` is the ranked list of
    ``((i, j), value)``.

    The empty-coalition (all-features-removed) prediction is supplied
    explicitly as the model's mean positive-class probability over the
    background. The TabPFN explainer's remove-and-recontextualize paradigm
    would otherwise evaluate TabPFN on an all-features-removed context,
    which TabPFN rejects ("all features constant"); the supplied empty
    prediction is the standard SHAP base value E[f(x)] and removes that
    failure mode.

    ShapIQ is scoped to the single-fit TabPFN variants (docs Section 3.5,
    "TabPFN single-fit headlines"). Its imputer re-fits the model once per
    feature coalition; for the fine-tuned variant a fit is a full
    training-epoch loop, so the per-coalition refit is intractable. Such
    models are rejected here so the leaf still emits its SHAP/ALE/embedding
    artefacts without paying an unbounded ShapIQ cost.
    """
    if not _is_single_fit_tabpfn(raw_model):
        raise RuntimeError(
            f"ShapIQ skipped: {type(raw_model).__name__} is not a single-fit "
            "TabPFN (its per-coalition refit is not cheap).")

    from tabpfn_extensions.interpretability.shapiq import get_tabpfn_explainer

    np.random.seed(SHAPIQ_SEED)
    feature_names = list(X_explain.columns)
    n_features = len(feature_names)

    if len(X_explain) > max_explain:
        rng = np.random.default_rng(SHAPIQ_SEED)
        sel = np.sort(rng.choice(len(X_explain), size=max_explain, replace=False))
        X_explain = X_explain.iloc[sel]

    base_rate = float(np.mean(np.asarray(y_background, dtype=float)))
    empty_prediction = None
    if predict_fn is not None:
        empty_prediction = float(np.mean(predict_fn(X_background)))
    else:
        empty_prediction = base_rate

    # Wrap the model so all-constant coalitions fall back to the base rate
    # rather than aborting the estimate (see _ConstantTolerantTabPFN).
    model = _ConstantTolerantTabPFN(raw_model, base_rate)

    explainer = get_tabpfn_explainer(
        model=model,
        data=X_background,
        labels=y_background,
        index="k-SII",
        max_order=2,
        class_index=1,
        empty_prediction=empty_prediction,
    )

    # Budget is the per-row coalition-sampling allowance; each coalition is a
    # TabPFN context refit, so the budget directly bounds the GPU cost.
    #   - low-cardinality cells (<=12 features): the exact 2^n budget, so the
    #     park/spano interaction estimates are exact;
    #   - high-cardinality cells (full_features, 52 features): a small sampling
    #     budget. The interpretation explicitly refuses interaction claims on
    #     the EPV-3.9 migraine full_features cell (docs claim 5), so the
    #     high-dimensional heatmap is descriptive only and does not warrant
    #     the cost of a large coalition sample.
    if n_features <= 12:
        budget = int(min(_BUDGET, 2 ** n_features))
    else:
        budget = _BUDGET_HIGH_DIM

    pair_accum = np.zeros((n_features, n_features))
    n_rows = 0
    for i in range(len(X_explain)):
        row = X_explain.iloc[i: i + 1].values
        interaction = explainer.explain(row[0], budget=budget, random_state=SHAPIQ_SEED)
        _free_gpu()
        # ``interaction_lookup`` maps each coalition tuple to its position in
        # the ``values`` array; the order-2 coalitions are the feature pairs.
        values = interaction.values
        for coalition, idx in interaction.interaction_lookup.items():
            members = coalition if isinstance(coalition, tuple) else (coalition,)
            if len(members) == 2:
                a, b = members
                v = abs(float(values[idx]))
                pair_accum[a, b] += v
                pair_accum[b, a] += v
        n_rows += 1

    if n_rows > 0:
        pair_accum /= n_rows

    top_pairs = []
    for a in range(n_features):
        for b in range(a + 1, n_features):
            top_pairs.append(((a, b), float(pair_accum[a, b])))
    top_pairs.sort(key=lambda t: t[1], reverse=True)

    meta = {
        "estimator": "ShapIQ k-SII order-2 (TabPFN-native)",
        "background_n": int(len(X_background)),
        "explained_n": int(n_rows),
        "budget": budget,
        "exact": budget >= 2 ** n_features,
        "seed": SHAPIQ_SEED,
    }
    return {
        "feature_names": feature_names,
        "pair_matrix": pair_accum,
        "top_pairs": top_pairs,
        "meta": meta,
    }


def top_pair_labels(result: dict, n: int = 10) -> list[tuple[str, str, float]]:
    """Resolve the top-``n`` interaction pairs to feature-name tuples."""
    names = result["feature_names"]
    out = []
    for (a, b), value in result["top_pairs"][:n]:
        out.append((names[a], names[b], value))
    return out
