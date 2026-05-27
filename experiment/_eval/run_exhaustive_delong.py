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
    at the given (target, feature_set, ratio=70_30, split)."""
    # XGB family at experiment/0/...
    xgb_root = _EXP_ROOT / "0" / target / feature_set / "stacked_2xgb_meta_lr" / RATIO / split
    if (xgb_root / "NonHP").is_dir() and (xgb_root / "NonHP" / "model.joblib").is_file():
        yield ("xgb_NonHP", xgb_root / "NonHP")
    hp_root = xgb_root / "HyperparameterTuned" / "single_AUROC"
    if hp_root.is_dir():
        for hp_variant in sorted(hp_root.iterdir()):
            if hp_variant.is_dir() and (hp_variant / "model.joblib").is_file():
                yield (f"xgb_{hp_variant.name}", hp_variant)
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
    cells = []
    all_pvalues = []
    test_pointers = []  # (cell_idx, arch_label_within_cell) for each p-value

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
                    except Exception as e:
                        print(f"     EXTRACT FAIL {arch_label}: {type(e).__name__}: {e}")
                        continue
                    # AUROC via sklearn (cheaper than full DeLong components here)
                    from sklearn.metrics import roc_auc_score
                    try:
                        auc = float(roc_auc_score(y_true, y_score))
                    except Exception:
                        continue
                    arch_results.append(dict(arch=arch_label, auc=auc,
                                             y_true=y_true, y_score=y_score))

                if len(arch_results) < 2:
                    print(f"     skip: only {len(arch_results)} arch with predictions")
                    continue

                arch_results.sort(key=lambda d: -d["auc"])
                headline = arch_results[0]
                cell_idx = len(cells)
                cell_rows = []

                for r in arch_results:
                    if r is headline:
                        cell_rows.append(dict(
                            arch=r["arch"], auc=r["auc"], delta_vs_headline=0.0,
                            p_vs_headline=None, is_headline=True,
                        ))
                        continue
                    _, _, delta, p, _, _ = delong_paired_test(
                        headline["y_true"], headline["y_score"], r["y_score"]
                    )
                    cell_rows.append(dict(
                        arch=r["arch"], auc=r["auc"],
                        delta_vs_headline=float(r["auc"] - headline["auc"]),
                        p_vs_headline=float(p), is_headline=False,
                    ))
                    all_pvalues.append(float(p))
                    test_pointers.append((cell_idx, r["arch"]))

                cells.append(dict(
                    cell=cell_label, target=target, feature_set=feature_set,
                    split=split, ratio=RATIO,
                    headline_arch=headline["arch"], rows=cell_rows,
                ))

    qvalues = _bonferroni(np.array(all_pvalues), alpha=0.05) if all_pvalues else np.array([])
    for (cell_idx, arch_label), q in zip(test_pointers, qvalues):
        for row in cells[cell_idx]["rows"]:
            if row["arch"] == arch_label and not row["is_headline"]:
                row["q_bonferroni"] = float(q)
                row["sig_bonferroni"] = bool(q <= 0.05)
                break

    n_tests = len(all_pvalues)
    n_sig = int(sum(1 for q in qvalues if q <= 0.05))
    print(f"\nTotal: {len(cells)} cells, {n_tests} paired tests, {n_sig} significant after Bonferroni (α={0.05/max(n_tests,1):.2e})")

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = _THIS.parent / f"exhaustive_delong_{ts}.json"
    payload = dict(
        timestamp=ts,
        ratio=RATIO,
        n_cells=len(cells),
        n_tests=n_tests,
        n_significant_bonferroni=n_sig,
        bonferroni_alpha_star=0.05 / max(n_tests, 1),
        cells=cells,
    )
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Persisted to: {out_path.relative_to(_EXP_ROOT.parent)}")


if __name__ == "__main__":
    main()
