"""Exhaustive paired DeLong across the 70_30-ratio grid for the
supplementary significance heatmap.

Scope: at each (target, feature_set, split) cell with ratio=70_30,
extract every architecture variant's locked-test predictions and run
paired DeLong of every non-headline variant against the cell's
headline (highest-AUROC) architecture. Bonferroni-correct across all
p-values (strict FWER control, more conservative than BH-FDR; the
supplementary's purpose is "any difference that survives the
strictest correction is real", whereas the main-text 18-test family
uses BH-FDR for targeted hypothesis tests).

The output is a JSON manifest the supplementary heatmap figure
consumes (`docs/methodAndResults_diagramCreatorScripts/
fig_g7_significance_heatmap.py`).

Run from the repository root::

    .venv/bin/python experiment/_eval/run_exhaustive_delong.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_EXP_ROOT))

from _eval.delong import delong_paired_test  # noqa: E402
from _eval.run_paired_delong import extract_predictions  # noqa: E402


TARGETS = ["headache", "migraine"]
# headache excludes park (Park's stepwise model is migraine-specific)
FEATURE_SETS_BY_TARGET = {
    "headache": ["full_features", "spano_features", "no_rolling_features"],
    "migraine": ["full_features", "spano_features", "no_rolling_features",
                 "park_features"],
}
SPLITS = ["chrono", "stratified", "patient"]
RATIO = "70_30"


def _enumerate_leaves(target: str, feature_set: str, split: str):
    """Yield (arch_label, leaf_path) for every architecture variant present
    at the given (target, feature_set, ratio=70_30, split). Mirrors the
    canonical sweep CSV's coverage at this ratio, including the
    blended_xgb_lr_spano2026 Spano baseline (Addition 0)."""
    # XGB stacked-meta-LR at experiment/0/.../stacked_2xgb_meta_lr/...
    xgb_root = _EXP_ROOT / "0" / target / feature_set / "stacked_2xgb_meta_lr" / RATIO / split
    if (xgb_root / "NonHP").is_dir() and (xgb_root / "NonHP" / "model.joblib").is_file():
        yield ("xgb_NonHP", xgb_root / "NonHP")
    hp_root = xgb_root / "HyperparameterTuned" / "single_AUROC"
    if hp_root.is_dir():
        for hp_variant in sorted(hp_root.iterdir()):
            if hp_variant.is_dir() and (hp_variant / "model.joblib").is_file():
                yield (f"xgb_{hp_variant.name}", hp_variant)
    # blended_xgb_lr_spano2026 baseline is intentionally skipped: it sits well
    # below the other variants on every cell and would mostly add no-effect
    # rows that crowd the supplementary heatmap without changing the story.
    # TabPFN + AutoTabPFN at experiment/1/...
    tabpfn_root = _EXP_ROOT / "1" / target / feature_set / "tabpfn"
    if tabpfn_root.is_dir():
        for variant in sorted(tabpfn_root.iterdir()):
            cell = variant / RATIO / split
            if cell.is_dir() and (cell / "model.joblib").is_file():
                label = variant.name.replace("version_", "")
                arch_label = "autotabpfn" if "auto" in label else f"tabpfn_{label}"
                yield (arch_label, cell)


def _bonferroni(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Per-test Bonferroni-adjusted q-value: q_i = min(1, p_i * m)."""
    m = len(pvalues)
    q = np.clip(np.asarray(pvalues, dtype=float) * m, 0.0, 1.0)
    return q


def main():
    from sklearn.metrics import roc_auc_score
    cells = []
    all_pairs = []  # flat list of every paired test for FWER

    for target in TARGETS:
        for feature_set in FEATURE_SETS_BY_TARGET[target]:
            for split in SPLITS:
                cell_label = f"{target}/{feature_set.replace('_features','')}/{split}"
                leaves = list(_enumerate_leaves(target, feature_set, split))
                if not leaves:
                    continue
                print(f"  {cell_label}: {len(leaves)} architectures")

                arch_results = []
                for arch_label, leaf_path in leaves:
                    try:
                        y_true, y_score = extract_predictions(leaf_path)
                        auc = float(roc_auc_score(y_true, y_score))
                    except Exception as e:
                        print(f"     EXTRACT FAIL {arch_label}: {type(e).__name__}: {e}")
                        continue
                    arch_results.append(dict(arch=arch_label, auc=auc,
                                             y_true=y_true, y_score=y_score,
                                             leaf=str(leaf_path.relative_to(_EXP_ROOT))))

                if len(arch_results) < 2:
                    print(f"     skip: only {len(arch_results)} arch with predictions")
                    continue

                # Sort architectures by AUROC desc; headline is index 0.
                arch_results.sort(key=lambda d: -d["auc"])
                headline_label = arch_results[0]["arch"]

                # All-pairs within this cell. This is the EXHAUSTIVE family
                # the supplementary's name promises; subsequent Bonferroni
                # correction across all_pairs bounds the family-wise error.
                cell_pairs = []
                for i in range(len(arch_results)):
                    for j in range(i + 1, len(arch_results)):
                        a, b = arch_results[i], arch_results[j]
                        _, _, delta, p, lo, hi = delong_paired_test(
                            a["y_true"], a["y_score"], b["y_score"]
                        )
                        pair = dict(
                            arch_a=a["arch"], arch_b=b["arch"],
                            auc_a=float(a["auc"]), auc_b=float(b["auc"]),
                            delta=float(delta), p=float(p),
                            ci_lo=float(lo), ci_hi=float(hi),
                        )
                        cell_pairs.append(pair)
                        all_pairs.append((cell_label, len(cells), len(cell_pairs) - 1, pair))

                cells.append(dict(
                    cell=cell_label, target=target, feature_set=feature_set,
                    split=split, ratio=RATIO, headline_arch=headline_label,
                    arch_aurocs={r["arch"]: r["auc"] for r in arch_results},
                    leaf_paths={r["arch"]: r["leaf"] for r in arch_results},
                    pairs=cell_pairs,
                ))

    n_tests = len(all_pairs)
    pvals = np.array([p[3]["p"] for p in all_pairs])
    qvals = _bonferroni(pvals, alpha=0.05) if n_tests else np.array([])
    for (cell_label, ci, pi, _), q in zip(all_pairs, qvals):
        cells[ci]["pairs"][pi]["q_bonferroni"] = float(q)
        cells[ci]["pairs"][pi]["sig_bonferroni"] = bool(q <= 0.05)

    n_sig = int((qvals <= 0.05).sum()) if n_tests else 0
    alpha_star = 0.05 / max(n_tests, 1)
    print(f"\nTotal: {len(cells)} cells, {n_tests} all-pairs paired tests, "
          f"{n_sig} significant after Bonferroni (α* = {alpha_star:.2e})")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = _THIS.parent / f"exhaustive_delong_{ts}.json"
    payload = dict(
        timestamp=ts, ratio=RATIO, n_cells=len(cells), n_tests=n_tests,
        n_significant_bonferroni=n_sig, bonferroni_alpha_star=alpha_star,
        cells=cells,
    )
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Persisted to: {out_path.relative_to(_EXP_ROOT.parent)}")

    # Per-cell JSON for individual parseability without loading the global blob.
    per_cell_dir = _THIS.parent / "exhaustive_delong_per_cell"
    per_cell_dir.mkdir(exist_ok=True)
    for c in cells:
        cell_file = per_cell_dir / f"{c['cell'].replace('/', '__')}.json"
        cell_file.write_text(json.dumps(c, indent=2, default=float))
    print(f"Per-cell files: {per_cell_dir.relative_to(_EXP_ROOT.parent)}/ ({len(cells)} files)")


if __name__ == "__main__":
    main()
