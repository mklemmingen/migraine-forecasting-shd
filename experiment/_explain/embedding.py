"""2D projection of TabPFN in-context attention embeddings.

For TabPFN leaves the in-context-learning attention embedding of each row
is projected to 2D and coloured by predicted probability, showing whether
the model has learned a neighbourhood structure that separates
migraine-day clusters. This is descriptive, not inferential, and is
omitted for the XGBoost leaves (no comparable embedding).

The embedding is read via the package's own accessor
(``raw_model.get_embeddings`` on a fitted TabPFN, surfaced by
``tabpfn_extensions.embedding.TabPFNEmbedding``) so no private model
internals are touched.

UMAP [mcinnes2018umap] is the primary projection: the figure's question is
cluster separation, and UMAP preserves global / inter-cluster structure
better than t-SNE and is far more layout-stable across re-runs with a fixed
``random_state``. t-SNE (scikit-learn, no new dependency) is the documented
fallback if a ``umap-learn`` import fails. Either projection distorts
distance, so the caption carries an explicit over-interpretation caveat
[wattenberg2016tsne, chari2023specious].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EMBED_SEED = 42
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


def _extract_embeddings(raw_model, X_explain: pd.DataFrame) -> np.ndarray | None:
    """Pull the per-row attention embedding from a fitted TabPFN model.

    The loaded ``model.joblib`` is already fitted, so ``get_embeddings`` is
    called directly with ``data_source='test'``. Returns ``None`` when the
    model exposes no embedding accessor (e.g. an AutoTabPFN handle), so the
    caller can skip the figure cleanly.
    """
    if not hasattr(raw_model, "get_embeddings"):
        return None
    try:
        emb = raw_model.get_embeddings(X_explain, data_source="test")
    except Exception:
        return None
    emb = np.asarray(emb)
    # TabPFN returns (n_estimators, n_samples, embedding_dim); average over
    # the ensemble axis to a single per-row vector when 3D.
    if emb.ndim == 3:
        emb = emb.mean(axis=0)
    if emb.ndim != 2:
        return None
    return emb


def _project(embeddings: np.ndarray) -> tuple[np.ndarray, str, dict]:
    """Project to 2D with UMAP; fall back to t-SNE if UMAP is unavailable."""
    n = len(embeddings)
    try:
        import umap

        n_neighbors = min(UMAP_N_NEIGHBORS, max(2, n - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=UMAP_MIN_DIST,
            random_state=EMBED_SEED,
        )
        coords = reducer.fit_transform(embeddings)
        meta = {"method": "UMAP", "n_neighbors": n_neighbors, "min_dist": UMAP_MIN_DIST,
                "random_state": EMBED_SEED}
        return coords, "UMAP", meta
    except Exception:
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=EMBED_SEED, init="pca")
        coords = reducer.fit_transform(embeddings)
        meta = {"method": "t-SNE (fallback)", "perplexity": perplexity, "random_state": EMBED_SEED}
        return coords, "t-SNE", meta


def compute_embedding_projection(
    raw_model,
    X_explain: pd.DataFrame,
    predicted_proba: np.ndarray,
) -> dict | None:
    """Return ``{coords, proba, method, meta}`` for the embedding figure,
    or ``None`` when the model has no embedding accessor."""
    embeddings = _extract_embeddings(raw_model, X_explain)
    if embeddings is None:
        return None
    coords, method, meta = _project(embeddings)
    return {
        "coords": coords,
        "proba": np.asarray(predicted_proba),
        "method": method,
        "meta": meta,
    }
