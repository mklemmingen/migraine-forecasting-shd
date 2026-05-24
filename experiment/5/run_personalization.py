"""Driver for the personalisation + within-person evaluation layer (Addition 5).

Implements the first, highest-value piece (docs Section 6, step 1): the
within-person evaluation over the EXISTING Additions 0/1/4 leaves, with zero new
training. For each selected leaf a fresh subprocess (_personal/_predict_worker.py)
regenerates the val+test predictions while retaining patient_id - run in its own
interpreter so the conflicting ``_model_architecture`` packages (exp/0 vs exp/4)
and the differing devices (TabPFN on GPU, XGBoost/sequence on CPU) never collide.
This driver then computes the per-patient AUROC/AUPRC distribution, the
precision-weighted random-effects within-person C-statistic, and the
pooled-vs-within gap (RQ2), across architectures.

The personalisation REGIMES (per-patient, partial-pooling) and the cold-start
curve are the next step (see _personal/regimes.py, _personal/walkforward.py).
Design and decisions: docs/addition5_personalization.md.
"""
import datetime as _dt
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent          # experiment/5/
EXP = HERE.parent                               # experiment/
sys.path[0:0] = [str(EXP), str(HERE), str(HERE / "_personal")]
import within_person as WP  # noqa: E402

RATIOS = ("70_15_15", "70_30", "80_20")
SPLITS = ("chrono", "stratified", "patient")
WORKER = HERE / "_personal" / "_predict_worker.py"       # hold-out (val+test)
CV_WORKER = HERE / "_personal" / "_cv_oof_worker.py"     # CV out-of-fold (refit per fold)


def _leaf_dims(model_dir: Path) -> dict:
    parts = model_dir.relative_to(EXP).parts
    return {"addition": parts[0], "target": parts[1], "feature_set": parts[2],
            "architecture": parts[3] if len(parts) > 3 else "?",
            "ratio": next((p for p in parts if p in RATIOS), "?"),
            "split": next((p for p in parts if p in SPLITS), "?")}


def _worker_env(addition: str) -> dict:
    """GPU visible only for TabPFN (Addition 1); hidden (CPU) for the XGBoost
    stack and the sequence model, which are CPU-native / CPU-saved here."""
    env = os.environ.copy()
    if addition != "1":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
    return env


def regenerate_predictions(model_dir: Path, worker: Path = WORKER):
    """Subprocess the given worker for one leaf; return (y, p, patient_id, dims).
    CV-OOF refits per fold, so it is given a long timeout."""
    d = _leaf_dims(model_dir)
    out = Path(tempfile.gettempdir()) / f"wp_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(worker), str(model_dir), str(out)],
                           capture_output=True, text=True, timeout=3600,
                           env=_worker_env(d["addition"]))
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr.strip().splitlines() or ["worker failed"])[-1]
            raise RuntimeError(tail)
        z = np.load(out)
        return z["y"], z["p"], z["pid"], d
    finally:
        out.unlink(missing_ok=True)


def within_person_for_leaf(model_dir: Path, worker: Path = WORKER, min_pos: int = 3) -> dict:
    y, p, pid, d = regenerate_predictions(model_dir, worker=worker)
    scores = WP.per_patient_scores(y, p, pid, min_pos=min_pos)
    within = WP.within_person_cstatistic(scores)
    pooled = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    tag = "cv" if worker == CV_WORKER else "holdout"
    out_dir = HERE / d["target"] / d["feature_set"] / "pooled" / d["ratio"] / d["split"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    scores.to_csv(out_dir / f"within_person_{tag}_add{d['addition']}_{d['architecture']}_{ts}.csv",
                  index=False)
    return {"addition": d["addition"], "target": d["target"],
            "architecture": d["architecture"], "n_patients": len(scores),
            "k_estimable": within["k_estimable"], "pooled_auroc": pooled,
            "within_person": within["estimate"], "within_ci_low": within["ci_low"],
            "within_ci_high": within["ci_high"], "tau2": within["tau2"],
            "pooled_minus_within": pooled - within["estimate"]}


def default_leaves() -> list[Path]:
    """Cross-architecture set: the chronological full_features 70/15/15 headline
    leaf per (target, architecture-family) across Additions 0 (XGBoost stack),
    1 (TabPFN v3-default) and 4 (sequence window-MLP)."""
    leaves = []
    for tgt in ("headache", "migraine"):
        # Addition 0 - XGBoost stack, NonHP
        for m in (EXP / "0" / tgt / "full_features").rglob("NonHP/model.joblib"):
            dd = _leaf_dims(m.parent)
            if dd["ratio"] == "70_15_15" and dd["split"] == "chrono" and "stacked_2xgb" in str(m):
                leaves.append(m.parent)
                break
        # Additions 1 (TabPFN) and 4 (sequence)
        for sub in ("1/{t}/full_features/tabpfn/version_3-default/70_15_15/chrono",
                    "4/{t}/full_features/sequence/version_window-mlp/70_15_15/chrono"):
            d = EXP / sub.format(t=tgt)
            if (d / "model.joblib").exists():
                leaves.append(d)
    return leaves


def main(leaves=None, cv: bool = False):
    leaves = leaves or default_leaves()
    worker = CV_WORKER if cv else WORKER
    # CV out-of-fold gives each patient predictions across the full date range, so
    # the principled 5-positive floor becomes reachable; the hold-out is sparser.
    min_pos = 5 if cv else 3
    mode = "CV out-of-fold (refit per fold)" if cv else "hold-out (val+test)"
    print(f"Within-person evaluation - {mode} - over {len(leaves)} leaf(s) (min_pos={min_pos})")
    rows = []
    for leaf in leaves:
        try:
            r = within_person_for_leaf(leaf, worker=worker, min_pos=min_pos)
            rows.append(r)
            wp = (f"{r['within_person']:.3f} [{r['within_ci_low']:.3f}-{r['within_ci_high']:.3f}]"
                  if r["within_person"] == r["within_person"] else "n/a")
            print(f"  add{r['addition']} {r['target']:<8} {r['architecture']:<18} "
                  f"pooled {r['pooled_auroc']:.3f} | within {wp} "
                  f"(k={r['k_estimable']}/{r['n_patients']}) | gap {r['pooled_minus_within']:+.3f}")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  SKIP {leaf.relative_to(EXP)}: {type(e).__name__}: {e}")
    if rows:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = HERE / f"within_person_summary_{'cv' if cv else 'holdout'}_{ts}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved summary: {out}")


if __name__ == "__main__":
    main(cv="--cv" in sys.argv)
