"""Per-leaf text summary writer for the insight pass.

Writes ``explain_<ts>.txt`` into the leaf's ``insights/`` folder: the
top-10 features by mean absolute attribution, the ALE net slopes for the
top features, the top interaction pairs (TabPFN headlines only), and the
estimator metadata (background-sample count, seed, estimator class) so the
attribution is reproducible and the calibrated-probability KernelSHAP
choice is on record.
"""
from __future__ import annotations

from pathlib import Path


def write_summary(
    insights_dir: Path,
    timestamp: str,
    title: str,
    arch_family: str,
    ranking: list[tuple[str, float]],
    attribution_meta: dict,
    ale_slopes: list[tuple[str, float]],
    interaction_pairs: list[tuple[str, str, float]] | None,
    interaction_meta: dict | None,
    embedding_meta: dict | None,
) -> Path:
    """Render and write the per-leaf explanation text file. Returns its path."""
    is_perm = attribution_meta.get("is_permutation", False)
    metric_label = "mean |importance|" if is_perm else "mean |SHAP|"

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"EXPLAINABILITY SUMMARY - {title}")
    lines.append("=" * 70)
    lines.append(f"architecture family : {arch_family}")
    lines.append(f"attribution         : {attribution_meta.get('estimator', '?')}")
    bn = attribution_meta.get("background_n")
    if bn is not None:
        lines.append(f"background samples  : {bn} "
                     f"(val/cal set in evaluate scope; never the train set)")
    sk = attribution_meta.get("background_summary_k")
    if sk is not None:
        kind = attribution_meta.get("background_summary", "k-means")
        lines.append(f"background summary  : {kind} with k={sk}")
    lines.append(f"explained rows      : {attribution_meta.get('explained_n', '?')}")
    lines.append(f"seed                : {attribution_meta.get('seed', '?')}")
    if is_perm:
        lines.append(f"base AUROC          : {attribution_meta.get('base_auroc', float('nan')):.4f}")
        lines.append(f"permutation repeats : {attribution_meta.get('n_repeats', '?')}")
    lines.append("-" * 70)

    lines.append(f"Top features by {metric_label}:")
    for rank, (feat, val) in enumerate(ranking[:10], 1):
        lines.append(f"  {rank:>2}. {feat:<32} {val:.6f}")
    lines.append("-" * 70)

    if ale_slopes:
        lines.append("ALE net slope (last - first centred effect) for top features:")
        lines.append("  positive slope: feature increases predicted migraine probability")
        for feat, slope in ale_slopes:
            lines.append(f"  {feat:<32} {slope:+.6f}")
        lines.append("-" * 70)

    if interaction_pairs:
        lines.append("Top pairwise interactions (ShapIQ k-SII order-2, "
                     "mean |interaction|):")
        for fa, fb, val in interaction_pairs[:10]:
            lines.append(f"  {fa:<24} x {fb:<24} {val:.6f}")
        if interaction_meta is not None:
            lines.append(f"  [background n={interaction_meta.get('background_n', '?')}, "
                         f"explained n={interaction_meta.get('explained_n', '?')}, "
                         f"seed={interaction_meta.get('seed', '?')}]")
        lines.append("-" * 70)

    if embedding_meta is not None:
        lines.append(f"Embedding projection: {embedding_meta.get('method', '?')} "
                     f"{ {k: v for k, v in embedding_meta.items() if k != 'method'} }")
        lines.append("  Caveat: 2D embedding distances are a qualitative aid only; "
                     "inter-cluster distances must not be over-read.")
        lines.append("-" * 70)

    insights_dir.mkdir(parents=True, exist_ok=True)
    out_path = insights_dir / f"explain_{timestamp}.txt"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def write_ranking_csv(
    insights_dir: Path,
    timestamp: str,
    ranking: list[tuple[str, float]],
    metric_label: str,
) -> Path:
    """Write the complete feature ranking (not just the top-10) as a CSV.

    The text summary truncates to the top-10 for readability; the
    cross-leaf comparison and the Park-OR / prodromal-contamination checks
    need every feature's attribution, so the full ranking is persisted
    here as ``ranking_<ts>.csv`` with columns ``rank,feature,value``.
    """
    insights_dir.mkdir(parents=True, exist_ok=True)
    out_path = insights_dir / f"ranking_{timestamp}.csv"
    lines = [f"# metric: {metric_label}", "rank,feature,value"]
    for rank, (feat, val) in enumerate(ranking, 1):
        lines.append(f"{rank},{feat},{val:.8f}")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
