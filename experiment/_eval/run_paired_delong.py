"""Paired DeLong AUROC comparison with BH-FDR correction across the
cell-by-split groups.

The current cross-architecture comparison in `results_findings.md` rests
on overlapping bootstrap CIs ("the architectures are close on
chronological headache"), an informal inferential procedure. Paired
DeLong [delong1988comparing] tests the difference of two AUROCs on the
SAME labels and risk set, accounting for the structural covariance
between paired Mann-Whitney U-statistics via the Sun-Xu midrank
algorithm [sun2014fast]. BH-FDR [benjamini1995controlling] controls the
expected proportion of false positives among rejections across the
multi-cell test family.

The contrast list defined below contains 18 paired tests and is the
frozen analysis-plan scope referenced in the Methods section.

Per-instance test predictions are not persisted by the leaf evaluate.py
templates, so this script reloads each leaf's `model.joblib`,
reconstructs the deterministic test split via the same loader the leaf
uses, and predicts. The reconstructed (y_true, y_score) pair is then
fed to `delong_paired_test`.

Outputs a markdown table to stdout with columns
(cell, split, ratio, arch_a, arch_b, AUROC_a, AUROC_b, delta, p, q,
significant_at_q=0.05).

Run from the repository root with the project venv activated::

    .venv/bin/python experiment/_eval/run_paired_delong.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_EXP_ROOT))

from _dataRead.read import load_and_prep_data  # noqa: E402
from _eval.delong import delong_paired_test  # noqa: E402


def _get_loader(feature_set: str):
    """Return the column-filter loader appropriate for the feature set."""
    if feature_set == "full_features":
        return None
    if feature_set == "no_rolling_features":
        from _dataRead.filter_to_no_rolling_features import (
            select_non_rolling_features,
        )
        return select_non_rolling_features
    if feature_set == "park_features":
        from _dataRead.filter_to_park_features import select_park_features
        return select_park_features
    if feature_set == "spano_features":
        from _dataRead.filter_to_spano_features import select_spano_features
        return select_spano_features
    raise ValueError(f"unknown feature_set: {feature_set!r}")


def _predict_xgb(bundle, X):
    """XGB calibrated-probability prediction via the bundle interface."""
    addition_root = str(_EXP_ROOT / "0")
    if addition_root not in sys.path:
        sys.path.insert(0, addition_root)
    from _model_architecture.stacked_2xgb_meta_lr_hp.model import (
        calibrated_proba,
    )
    return np.asarray(calibrated_proba(bundle, X), dtype=float)


def _predict_tabpfn(model, X):
    """TabPFN / AutoTabPFN sklearn-style predict_proba positive column."""
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def extract_predictions(leaf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load model.joblib, reconstruct test split, return (y_true, y_score).

    The split reconstruction is deterministic: each test parquet is
    written once during the data-engineering pass and never mutated, so
    re-loading + re-predicting reproduces the exact instances the leaf
    evaluate.py used.
    """
    leaf = Path(leaf_path).resolve()
    rel = leaf.relative_to(_EXP_ROOT)
    addition = rel.parts[0]  # "0" or "1"
    target = rel.parts[1]
    feature_set = rel.parts[2]

    ratio = next(p for p in leaf.parts if p in {"70_30", "70_15_15", "80_20"})
    split_type = next(
        p for p in leaf.parts if p in {"chrono", "stratified", "patient"}
    )

    data_dir = _EXP_ROOT.parent / "data" / "processed" / target
    test_path = data_dir / ratio / split_type / "diary_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"test parquet missing: {test_path}")

    loader = _get_loader(feature_set)
    X_test, y_test = load_and_prep_data(str(test_path), loader=loader)

    model_path = leaf / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib missing: {model_path}")
    # Each addition root carries its own _model_architecture/ subpackage with
    # different submodules (XGB stacks under experiment/0; TabPFN, AutoTabPFN
    # variants under experiment/1). After importing one, sys.modules caches
    # the package binding, so a subsequent load from a different addition
    # cannot resolve its custom estimator classes. Evict the cache and put
    # the target addition root at position 0 before joblib.load.
    addition_root = str(_EXP_ROOT / addition)
    for cached in [m for m in sys.modules if
                   m == "_model_architecture" or m.startswith("_model_architecture.")]:
        del sys.modules[cached]
    while addition_root in sys.path:
        sys.path.remove(addition_root)
    sys.path.insert(0, addition_root)
    model = joblib.load(model_path)

    if addition == "0":
        y_score = _predict_xgb(model, X_test)
    elif addition == "1":
        y_score = _predict_tabpfn(model, X_test)
    else:
        raise ValueError(f"unknown addition: {addition}")

    return np.asarray(y_test).astype(int), y_score


@dataclass
class PairedTest:
    cell: str           # human-readable cell label, e.g. "headache/full/chrono"
    ratio: str
    arch_a_label: str
    arch_b_label: str
    leaf_a: Path
    leaf_b: Path


# 18 paired tests across five categories:
#   6 cross-family pairs at 70_30 canonical headline cells (headache
#     full/chrono, headache no_rolling/chrono, migraine full/chrono,
#     migraine park/chrono, migraine park/patient, migraine no_rolling
#     /patient);
#   3 within-family-tie pairs at headache full/chrono/70_30 where the
#     composite_sorted rule places v2.6, v3-default, and v3-binary in a
#     0.02-AUROC tie, with DeLong testing whether the tie is statistically
#     defensible;
#   3 cross-family pairs at 70_15_15 ratio (headache no_rolling/stratified,
#     migraine full/patient, migraine full/stratified);
#   4 HP-ladder pairs at migraine full/chrono/70_30 testing HP020 against
#     HP050, HP100, HP200, and HP500;
#   2 AutoTabPFN load-bearing pairs at migraine full/chrono/70_30 against
#     XGB-HP020 and XGB-NonHP.
ALL_PAIRS: list[PairedTest] = [
    # --- Cross-family pairs (same ratio, different architecture family) ---
    PairedTest(
        cell="headache/full/chrono",
        ratio="70_30",
        arch_a_label="tabpfn_v2-6",
        arch_b_label="stacked_2xgb_NonHP",
        leaf_a=_EXP_ROOT / "1/headache/full_features/tabpfn/version_2-6/70_30/chrono",
        leaf_b=_EXP_ROOT / "0/headache/full_features/stacked_2xgb_meta_lr/70_30/chrono/NonHP",
    ),
    PairedTest(
        cell="headache/no_rolling/chrono",
        ratio="70_30",
        arch_a_label="tabpfn_v3-binary",
        arch_b_label="stacked_2xgb_NonHP",
        leaf_a=_EXP_ROOT / "1/headache/no_rolling_features/tabpfn/version_3-binary/70_30/chrono",
        leaf_b=_EXP_ROOT / "0/headache/no_rolling_features/stacked_2xgb_meta_lr/70_30/chrono/NonHP",
    ),
    PairedTest(
        cell="migraine/full/chrono",
        ratio="70_30",
        arch_a_label="stacked_2xgb_HP020",
        arch_b_label="tabpfn_v2-5-finetuned",
        leaf_a=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP020",
        leaf_b=_EXP_ROOT / "1/migraine/full_features/tabpfn/version_2-5-finetuned/70_30/chrono",
    ),
    PairedTest(
        cell="migraine/park/chrono",
        ratio="70_30",
        arch_a_label="tabpfn_v3-binary",
        arch_b_label="stacked_2xgb_NonHP",
        leaf_a=_EXP_ROOT / "1/migraine/park_features/tabpfn/version_3-binary/70_30/chrono",
        leaf_b=_EXP_ROOT / "0/migraine/park_features/stacked_2xgb_meta_lr/70_30/chrono/NonHP",
    ),
    PairedTest(
        cell="migraine/park/patient",
        ratio="70_30",
        arch_a_label="stacked_2xgb_NonHP",
        arch_b_label="tabpfn_v3-default",
        leaf_a=_EXP_ROOT / "0/migraine/park_features/stacked_2xgb_meta_lr/70_30/patient/NonHP",
        leaf_b=_EXP_ROOT / "1/migraine/park_features/tabpfn/version_3-default/70_30/patient",
    ),
    PairedTest(
        cell="migraine/no_rolling/patient",
        ratio="70_30",
        arch_a_label="tabpfn_v3-binary",
        arch_b_label="stacked_2xgb_NonHP",
        leaf_a=_EXP_ROOT / "1/migraine/no_rolling_features/tabpfn/version_3-binary/70_30/patient",
        leaf_b=_EXP_ROOT / "0/migraine/no_rolling_features/stacked_2xgb_meta_lr/70_30/patient/NonHP",
    ),
    # --- Within-family-tie pairs at headache/full/chrono/70_30 ---
    PairedTest(
        cell="headache/full/chrono",
        ratio="70_30",
        arch_a_label="tabpfn_v2-6",
        arch_b_label="tabpfn_v3-default",
        leaf_a=_EXP_ROOT / "1/headache/full_features/tabpfn/version_2-6/70_30/chrono",
        leaf_b=_EXP_ROOT / "1/headache/full_features/tabpfn/version_3-default/70_30/chrono",
    ),
    PairedTest(
        cell="headache/full/chrono",
        ratio="70_30",
        arch_a_label="tabpfn_v2-6",
        arch_b_label="tabpfn_v3-binary",
        leaf_a=_EXP_ROOT / "1/headache/full_features/tabpfn/version_2-6/70_30/chrono",
        leaf_b=_EXP_ROOT / "1/headache/full_features/tabpfn/version_3-binary/70_30/chrono",
    ),
    PairedTest(
        cell="headache/full/chrono",
        ratio="70_30",
        arch_a_label="tabpfn_v3-default",
        arch_b_label="tabpfn_v3-binary",
        leaf_a=_EXP_ROOT / "1/headache/full_features/tabpfn/version_3-default/70_30/chrono",
        leaf_b=_EXP_ROOT / "1/headache/full_features/tabpfn/version_3-binary/70_30/chrono",
    ),
    # --- Agent 1's 3 missing cross-family pairs at 70_15_15 ratio ---
    PairedTest(
        cell="headache/no_rolling/stratified",
        ratio="70_15_15",
        arch_a_label="tabpfn_v2-5-real",
        arch_b_label="stacked_2xgb_NonHP",
        leaf_a=_EXP_ROOT / "1/headache/no_rolling_features/tabpfn/version_2-5-real/70_15_15/stratified",
        leaf_b=_EXP_ROOT / "0/headache/no_rolling_features/stacked_2xgb_meta_lr/70_15_15/stratified/NonHP",
    ),
    PairedTest(
        cell="migraine/full/patient",
        ratio="70_15_15",
        arch_a_label="tabpfn_v3-binary",
        arch_b_label="stacked_2xgb_HP020",
        leaf_a=_EXP_ROOT / "1/migraine/full_features/tabpfn/version_3-binary/70_15_15/patient",
        leaf_b=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_15_15/patient/HyperparameterTuned/single_AUROC/HP020",
    ),
    PairedTest(
        cell="migraine/full/stratified",
        ratio="70_15_15",
        arch_a_label="tabpfn_v2-5-finetuned",
        arch_b_label="autotabpfn_v2-5-auto",
        leaf_a=_EXP_ROOT / "1/migraine/full_features/tabpfn/version_2-5-finetuned/70_15_15/stratified",
        leaf_b=_EXP_ROOT / "1/migraine/full_features/tabpfn/version_2-5-auto/70_15_15/stratified",
    ),
    # --- HP-ladder at migraine/full/chrono/70_30: does HP020 actually beat
    #     higher-budget search? The composite-sorted winner is HP020; this
    #     ladder tests whether the within-budget paired difference is
    #     statistically resolvable.
    PairedTest(
        cell="migraine/full/chrono",
        ratio="70_30",
        arch_a_label="stacked_2xgb_HP020",
        arch_b_label="stacked_2xgb_HP050",
        leaf_a=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP020",
        leaf_b=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP050",
    ),
    PairedTest(
        cell="migraine/full/chrono",
        ratio="70_30",
        arch_a_label="stacked_2xgb_HP020",
        arch_b_label="stacked_2xgb_HP100",
        leaf_a=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP020",
        leaf_b=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP100",
    ),
    PairedTest(
        cell="migraine/full/chrono",
        ratio="70_30",
        arch_a_label="stacked_2xgb_HP020",
        arch_b_label="stacked_2xgb_HP200",
        leaf_a=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP020",
        leaf_b=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP200",
    ),
    PairedTest(
        cell="migraine/full/chrono",
        ratio="70_30",
        arch_a_label="stacked_2xgb_HP020",
        arch_b_label="stacked_2xgb_HP500",
        leaf_a=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP020",
        leaf_b=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP500",
    ),
    # --- AutoTabPFN load-bearing tests at migraine/full/chrono/70_30: the
    #     paper's Results-section claim "AutoTabPFN is the migraine leader" needs paired
    #     tests against both XGB-HP020 (composite winner) and XGB-NonHP
    #     (the Results section's numeric comparison anchor at 0.745 vs 0.660).
    PairedTest(
        cell="migraine/full/chrono",
        ratio="70_30",
        arch_a_label="autotabpfn_v2-5-auto",
        arch_b_label="stacked_2xgb_HP020",
        leaf_a=_EXP_ROOT / "1/migraine/full_features/tabpfn/version_2-5-auto/70_30/chrono",
        leaf_b=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP020",
    ),
    PairedTest(
        cell="migraine/full/chrono",
        ratio="70_30",
        arch_a_label="autotabpfn_v2-5-auto",
        arch_b_label="stacked_2xgb_NonHP",
        leaf_a=_EXP_ROOT / "1/migraine/full_features/tabpfn/version_2-5-auto/70_30/chrono",
        leaf_b=_EXP_ROOT / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/NonHP",
    ),
]


def bh_fdr(pvalues: np.ndarray, q_target: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted q-values for the input p-values.

    Returns an array of monotone-corrected q-values; significance at the
    chosen q_target is `q <= q_target`. Implemented via the standard
    sort-rank-correct-unsort sequence; ties are handled by the sort.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * m / (np.arange(m) + 1)
    # enforce monotone non-increasing from the largest q downward
    q_monotone = np.minimum.accumulate(q_ranked[::-1])[::-1]
    out = np.empty_like(q_monotone)
    out[order] = np.clip(q_monotone, 0.0, 1.0)
    return out


def run_pair(test: PairedTest) -> dict:
    """Extract predictions for both leaves and run paired DeLong."""
    y_a, s_a = extract_predictions(test.leaf_a)
    y_b, s_b = extract_predictions(test.leaf_b)
    if not np.array_equal(y_a, y_b):
        raise ValueError(
            f"y_true mismatch between {test.leaf_a} and {test.leaf_b}; "
            "the two leaves must share the test instances for paired DeLong"
        )
    auc_a, auc_b, delta, p_value, lo, hi = delong_paired_test(y_a, s_a, s_b)
    return dict(
        cell=test.cell,
        ratio=test.ratio,
        arch_a=test.arch_a_label,
        arch_b=test.arch_b_label,
        auc_a=auc_a,
        auc_b=auc_b,
        delta=delta,
        ci_lo=lo,
        ci_hi=hi,
        p=p_value,
    )


def main():
    print("Running paired DeLong tests across", len(ALL_PAIRS), "cell-pair groups...")
    rows = []
    for i, t in enumerate(ALL_PAIRS, 1):
        print(f"  [{i:2d}/{len(ALL_PAIRS)}] {t.cell}  {t.arch_a_label} vs {t.arch_b_label}")
        try:
            rows.append(run_pair(t))
        except Exception as e:
            print(f"     SKIPPED: {type(e).__name__}: {e}")
            continue

    if not rows:
        print("No tests ran successfully.")
        return

    pvalues = np.array([r["p"] for r in rows], dtype=float)
    qvalues = bh_fdr(pvalues, q_target=0.05)
    for r, q in zip(rows, qvalues):
        r["q"] = float(q)
        r["sig_at_q05"] = bool(q <= 0.05)

    print()
    print("=" * 100)
    print("PAIRED DELONG + BH-FDR (q=0.05)")
    print("=" * 100)
    header = (
        f"{'cell':28s} {'ratio':8s} {'arch_a':22s} {'arch_b':22s} "
        f"{'AUC_a':>6s} {'AUC_b':>6s} {'delta':>7s} {'p':>7s} {'q':>7s} sig"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        print(
            f"{r['cell']:28s} {r['ratio']:8s} {r['arch_a']:22s} {r['arch_b']:22s} "
            f"{r['auc_a']:.3f} {r['auc_b']:.3f} {r['delta']:+.3f}  "
            f"{r['p']:.4f} {r['q']:.4f} {'*' if r['sig_at_q05'] else ' '}"
        )
    print("-" * 100)
    n_sig = sum(1 for r in rows if r["sig_at_q05"])
    print(f"Significant at q ≤ 0.05: {n_sig} of {len(rows)} comparisons")
    print()
    print("Markdown table (paste-ready for results_findings.md):")
    print()
    print("| cell | ratio | arch A | arch B | AUROC A | AUROC B | ΔAUC | p (DeLong) | q (BH) | sig |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sig = "**yes**" if r["sig_at_q05"] else "no"
        print(
            f"| {r['cell']} | {r['ratio']} | {r['arch_a']} | {r['arch_b']} | "
            f"{r['auc_a']:.3f} | {r['auc_b']:.3f} | {r['delta']:+.3f} | "
            f"{r['p']:.4f} | {r['q']:.4f} | {sig} |"
        )

    # Persist to versioned JSON for later figure generation / re-analysis.
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = _THIS.parent / f"paired_delong_{ts}.json"
    payload = {
        "timestamp": ts,
        "script": str(_THIS.relative_to(_EXP_ROOT.parent)),
        "n_tests": len(rows),
        "n_significant_at_q05": n_sig,
        "q_target": 0.05,
        "tests": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nPersisted to: {out_path.relative_to(_EXP_ROOT.parent)}")


if __name__ == "__main__":
    main()
