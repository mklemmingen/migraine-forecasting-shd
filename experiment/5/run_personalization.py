"""Driver for the personalisation + within-person evaluation layer (Addition 5).

This implements the first, highest-value piece (docs Section 6, step 1): the
within-person evaluation over the EXISTING Additions 0/1/4 leaves, with zero new
training. For each selected leaf it loads model.joblib, regenerates the test
predictions while RETAINING patient_id (the column prep_split drops), and
computes the per-patient AUROC/AUPRC distribution, the precision-weighted
random-effects within-person C-statistic, and the pooled-vs-within gap (RQ2).

The personalisation REGIMES (per-patient, partial-pooling, TabPFN-in-context)
and the cold-start curve are the next step; their orchestration lives in
_personal/regimes.py and _personal/walkforward.py (TODOs), and they emit the
standard results contract so they fold into comparison_*.html. The pooled
baseline here reuses the existing leaf's metrics, so it emits only the
within-person artifact (re-emitting the standard results would duplicate the
0/1/4 row).

Design and decisions: docs/addition5_personalization.md.
"""
import datetime as _dt
import sys
import uuid
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent          # experiment/5/
EXP = HERE.parent                               # experiment/
REPO = EXP.parent
sys.path[0:0] = [str(EXP), str(HERE), str(HERE / "_personal")]
from _dataRead.read import load_raw, prep_split  # noqa: E402
from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
from _dataRead.filter_to_park_features import select_park_features  # noqa: E402
import within_person as WP  # noqa: E402

RATIOS = ("70_15_15", "70_30", "80_20")
SPLITS = ("chrono", "stratified", "patient")
LOADERS = {"full_features": None,
           "no_rolling_features": select_non_rolling_features,
           "park_features": select_park_features}


def _leaf_dims(model_dir: Path) -> dict:
    """Extract (addition, target, feature_set, architecture, ratio, split) from a
    leaf model directory path, robust to the variable NonHP/HP depth."""
    parts = model_dir.relative_to(EXP).parts
    ratio = next((p for p in parts if p in RATIOS), "?")
    split = next((p for p in parts if p in SPLITS), "?")
    return {"addition": parts[0], "target": parts[1], "feature_set": parts[2],
            "architecture": parts[3] if len(parts) > 3 else "?",
            "ratio": ratio, "split": split, "is_sequence": "sequence" in parts}


def _proba(bundle, d: dict, X):
    """Dispatch to the leaf's prediction API: Addition 0 uses the calibrated
    stacking closure ``calibrated_proba(bundle, X)``; Addition 1 (TabPFN) exposes
    sklearn ``predict_proba``. Sequence (Addition 4) is a follow-up."""
    if d["addition"] == "0":
        sys.path.insert(0, str(EXP / "0"))
        from _model_architecture.stacked_2xgb_meta_lr.model import calibrated_proba  # noqa: E402
        return calibrated_proba(bundle, X)
    return bundle.predict_proba(X)[:, 1]


def regenerate_predictions(model_dir: Path):
    """Load model_dir/model.joblib, predict on the leaf's test split, and return
    (y, p, patient_id, dims) row-aligned. patient_id is retained for grouping;
    load_raw keeps it while prep_split gives the model's feature matrix."""
    d = _leaf_dims(model_dir)
    if d["is_sequence"]:
        raise NotImplementedError("sequence-leaf prediction regeneration is a follow-up")
    # Evaluate on the model's full out-of-sample horizon: val + test. Both are
    # unseen by the train-fitted bundle (val is used only for threshold
    # selection downstream, not fitting), so combining them is leakage-free and
    # roughly doubles the per-patient days available for within-person estimation
    # on this chronological, late-enrolment-limited cohort. (The fully estimable
    # within-person evaluation uses CV out-of-fold predictions - the documented
    # follow-up, since the saved leaves do not persist per-fold predictions.)
    base = REPO / "data" / "processed" / d["target"] / d["ratio"] / d["split"]
    frames = [load_raw(str(base / f"diary_{sp}.parquet"), loader=LOADERS.get(d["feature_set"]))
              for sp in ("val", "test") if (base / f"diary_{sp}.parquet").exists()]
    df = pd.concat(frames, ignore_index=True)
    X, y = prep_split(df)
    bundle = joblib.load(model_dir / "model.joblib")
    p = _proba(bundle, d, X)
    return y.to_numpy(), p, df["patient_id"].to_numpy(), d


def within_person_for_leaf(model_dir: Path, min_pos: int = WP.MIN_POS) -> dict:
    """Per-patient within-person evaluation of one existing leaf's test split."""
    y, p, pid, d = regenerate_predictions(model_dir)
    scores = WP.per_patient_scores(y, p, pid, min_pos=min_pos)
    within = WP.within_person_cstatistic(scores)
    pooled = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    gap = WP.pooled_vs_within(pooled, within)

    out_dir = (HERE / d["target"] / d["feature_set"] / "pooled" / d["ratio"] / d["split"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    scores.to_csv(out_dir / f"within_person_{ts}_{uuid.uuid4().hex[:8]}.csv", index=False)
    return {**d, "n_patients": len(scores), "k_estimable": within["k_estimable"],
            "pooled_auroc": pooled, "within_person": within["estimate"],
            "within_ci_low": within["ci_low"], "within_ci_high": within["ci_high"],
            "tau2": within["tau2"], "median_patient_auroc": within["median_auroc"],
            "pooled_minus_within": gap["gap"]}


def default_leaves() -> list[Path]:
    """The canonical CPU-reliable demonstration set: the Addition 0 XGBoost-stack
    NonHP chronological full_features 70/15/15 leaf for each target (one per
    target; HP-tuned variants are excluded to keep the headline view clean)."""
    leaves = []
    for tgt in ("headache", "migraine"):
        for m in (EXP / "0" / tgt / "full_features").rglob("model.joblib"):
            d = _leaf_dims(m.parent)
            if (d["ratio"] == "70_15_15" and d["split"] == "chrono"
                    and "stacked_2xgb" in str(m) and m.parent.name == "NonHP"):
                leaves.append(m.parent)
    return leaves


def main(leaves=None, min_pos: int = 3):
    leaves = leaves or default_leaves()
    print(f"Within-person evaluation over {len(leaves)} existing leaf(s) (min_pos={min_pos})")
    rows = []
    for leaf in leaves:
        try:
            rows.append(within_person_for_leaf(leaf, min_pos=min_pos))
            r = rows[-1]
            print(f"  {r['addition']}/{r['target']}/{r['feature_set']}/{r['split']}: "
                  f"pooled AUROC {r['pooled_auroc']:.3f} | within-person "
                  f"{r['within_person']:.3f} [{r['within_ci_low']:.3f}-{r['within_ci_high']:.3f}] "
                  f"(k={r['k_estimable']}/{r['n_patients']}) | gap {r['pooled_minus_within']:+.3f}")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  SKIP {leaf.relative_to(EXP)}: {type(e).__name__}: {e}")
    if rows:
        summary = pd.DataFrame(rows)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = HERE / f"within_person_summary_{ts}.csv"
        summary.to_csv(out, index=False)
        print(f"\nSaved summary: {out}")


if __name__ == "__main__":
    main()
