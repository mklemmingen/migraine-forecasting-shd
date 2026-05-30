"""Patient-cluster bootstrap CIs on the headline-cell metrics cited in body
§3.2 (AUROC) / §3.4 (calibration slope + CITL) / §3.7 (Brier skill against
per-patient TRAIN climatology), side-by-side with the patient-day-iid CIs
currently in the body. Closes the T2-2 panel-revision ask (P4 Collins, P5
Riley, P6 Van Calster, P9 McElfresh, P10 Barnett) for the headline cells.

Resamples whole PATIENTS with replacement (the resampling unit is the
patient, not the patient-day row), then concatenates the resampled patients'
rows and recomputes the metric. This is the appropriate unit because the
within-patient day-level outcomes are not independent (Addition 3 quantified
substantial within-patient serial dependence), and patient-day-iid bootstrap
under-covers relative to patient-cluster when the substantive claim is
direction-of-effect rather than equivalence (Barnett's quote at body §3.7).

The existing patient-day CIs from `experiment/_eval/metrics_lib.py` +
`experiment/6/_value/skill.py` are preserved as side-by-side sensitivity
values per the §2.8 reporting convention: primary = patient-cluster,
sensitivity = patient-day for literature comparability.

Uses the hold-out test predict worker (NOT the cv_oof worker fig_c5 / T3-5 /
T3-6 / T2-6 used) because §3.2 / §3.4 / §3.7 cite TEST-SET metrics not OOF
metrics; the within-person C-statistic in §3.6 is the only metric that uses
OOF predictions and that estimator is already patient-level via
Hanley-McNeil + Paule-Mandel pooling rather than a row bootstrap, so the
within-person C is not retouched by this commit.

Usage: ``python experiment/_eval/_special/run_patient_cluster_bootstrap.py``
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
REPO = EXP.parent

sys.path[0:0] = [str(EXP)]
from _eval.metrics_lib import calibration_slope  # noqa: E402

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"

N_BOOT = 1000


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path):
    """Hold-out predict (val+test). Returns (y, p, pid)."""
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"pcb_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run(
            [sys.executable, str(WORKER), str(leaf), str(out)],
            capture_output=True, text=True, timeout=1800, env=_env(addition),
        )
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr.strip().splitlines() or ["worker failed"])[-1]
            print(f"  SKIP {leaf.name}: {tail}")
            return None
        z = np.load(out, allow_pickle=True)
        return (z["y"].astype(int), z["p"].astype(float), z["pid"].astype(str))
    finally:
        out.unlink(missing_ok=True)


def _train_climatology(target: str, ratio: str, split: str) -> tuple[dict, float]:
    """Per-patient train-set climatology (leakage-free reference for Brier
    skill) + cohort fallback for patients not seen in train.

    Per ``project_target_column_convention.md``, BOTH the headache/ and
    migraine/ processed parquets use the column name ``migraine_target`` -
    the column-name reuse is by-design and the actual target semantics live
    in the parent directory choice, not the column name. So the column key
    is constant; the parent directory in ``REPO / data / processed /
    target / ...`` is what differs."""
    target_col = "migraine_target"
    train_pq = REPO / "data" / "processed" / target / ratio / split / "diary_train.parquet"
    tr = pd.read_parquet(train_pq)
    rates = tr.groupby("patient_id")[target_col].mean()
    rates.index = rates.index.astype(str)
    return rates.to_dict(), float(tr[target_col].mean())


def _reference_array(pid: np.ndarray, rates: dict, cohort_rate: float) -> np.ndarray:
    """Map each row's patient_id to its train-set climatology; cohort base
    rate for patients not in the train set (the existing skill.py contract)."""
    return np.array([rates.get(str(p), cohort_rate) for p in pid], dtype=float)


def _auroc(y, p) -> float:
    if len(set(y)) <= 1:
        return float("nan")
    return float(roc_auc_score(y, p))


def _citl(y, p) -> float:
    mp = float(np.mean(p))
    return float(np.mean(y) / mp) if mp > 0 else float("nan")


def _brier_skill(y, p, ref) -> float:
    bs_model = float(np.mean((p - y) ** 2))
    bs_ref = float(np.mean((ref - y) ** 2))
    return 1.0 - bs_model / bs_ref if bs_ref > 0 else float("nan")


def _bootstrap(metric_fn, y, p, n_boot: int, seed: int,
               pid: np.ndarray | None = None, ref: np.ndarray | None = None):
    """If pid is None: row bootstrap (patient-day-iid; current §2.8 default).
    If pid is given: patient-cluster bootstrap (sample patient_ids with
    replacement, concat their rows). When ref is given, metric_fn takes
    (y, p, ref); otherwise (y, p)."""
    rng = np.random.default_rng(seed)
    vals = []

    if pid is None:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                args = (y[idx], p[idx], ref[idx]) if ref is not None else (y[idx], p[idx])
                v = metric_fn(*args)
                if v == v:  # not NaN
                    vals.append(v)
            except Exception:
                pass
    else:
        patients, inv = np.unique(pid, return_inverse=True)
        patient_rows = [np.where(inv == i)[0] for i in range(len(patients))]
        n_patients = len(patients)
        for _ in range(n_boot):
            sampled = rng.integers(0, n_patients, size=n_patients)
            idx = np.concatenate([patient_rows[s] for s in sampled])
            try:
                args = (y[idx], p[idx], ref[idx]) if ref is not None else (y[idx], p[idx])
                v = metric_fn(*args)
                if v == v:
                    vals.append(v)
            except Exception:
                pass

    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.array(vals)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), len(arr)


def _resolve_cells() -> list[tuple[str, str, Path]]:
    """Resolve the 6 cells cited in §3.7 (3 architectures × 2 targets) via the
    composite-tracked figdata (XGBoost + TabPFN) plus pinned window-MLP."""
    spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    fp = mod.latest_figdata(EXP / "2")
    if fp is None:
        raise SystemExit("no figdata_*.json in experiment/2/")
    print(f"  source {fp.name}")
    headlines = mod.load_figdata(fp).get("headlines", [])

    def find(tgt: str, family: str) -> Path | None:
        for role in ("headline", "runner_up"):
            for e in headlines:
                if (e.get("target") == tgt and e.get("family") == family
                        and e.get("feature_set") == "full_features"
                        and e.get("splittype") == "chrono"
                        and e.get("role") == role and e.get("leaf_dir")):
                    return Path(e["leaf_dir"])
        return None

    cells: list[tuple[str, str, Path]] = []
    for tgt in ("headache", "migraine"):
        for label, family in (("XGBoost", "xgboost"), ("TabPFN", "tabpfn")):
            d = find(tgt, family)
            if d is not None and (d / "model.joblib").exists():
                cells.append((tgt, label, d))
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            cells.append((tgt, "sequence", d4))
    return cells


def main() -> None:
    cells = _resolve_cells()
    print(f"Resolved {len(cells)} cells:")
    for tgt, label, leaf in cells:
        print(f"  {tgt:<9} {label:<10} {leaf.relative_to(EXP)}")
    print()

    rows: list[dict] = []
    for tgt, label, leaf in cells:
        t0 = time.time()
        print(f"=== {tgt} / {label} ===")
        r = _predict(leaf)
        if r is None:
            continue
        y, p, pid = r
        # Reverse-engineer ratio/split from path
        parts = leaf.relative_to(EXP).parts
        ratio = next((q for q in parts if q in ("70_30", "70_15_15", "80_20")), "70_30")
        split = next((q for q in parts if q in ("chrono", "stratified", "patient")), "chrono")
        rates, cohort = _train_climatology(tgt, ratio, split)
        ref = _reference_array(pid, rates, cohort)
        print(f"  predict {time.time()-t0:.0f}s  n_rows={len(y)}  n_patients={len(set(pid))}  ratio={ratio}/{split}")

        # Point estimates
        est_auroc = _auroc(y, p)
        est_slope = float(calibration_slope(y, p))
        est_citl = _citl(y, p)
        est_brier_skill = _brier_skill(y, p, ref)

        # Bootstrap CIs: patient-day (current §2.8) AND patient-cluster (T2-2 primary)
        auroc_day = _bootstrap(_auroc, y, p, N_BOOT, 42)
        auroc_cluster = _bootstrap(_auroc, y, p, N_BOOT, 42, pid=pid)
        slope_day = _bootstrap(lambda yy, pp: float(calibration_slope(yy, pp)), y, p, N_BOOT, 42)
        slope_cluster = _bootstrap(lambda yy, pp: float(calibration_slope(yy, pp)), y, p, N_BOOT, 42, pid=pid)
        citl_day = _bootstrap(_citl, y, p, N_BOOT, 42)
        citl_cluster = _bootstrap(_citl, y, p, N_BOOT, 42, pid=pid)
        bs_day = _bootstrap(_brier_skill, y, p, N_BOOT, 42, ref=ref)
        bs_cluster = _bootstrap(_brier_skill, y, p, N_BOOT, 42, pid=pid, ref=ref)

        print(f"  AUROC        est={est_auroc:.3f}  day [{auroc_day[0]:.3f},{auroc_day[1]:.3f}]  cluster [{auroc_cluster[0]:.3f},{auroc_cluster[1]:.3f}]")
        print(f"  slope        est={est_slope:.3f}  day [{slope_day[0]:.3f},{slope_day[1]:.3f}]  cluster [{slope_cluster[0]:.3f},{slope_cluster[1]:.3f}]")
        print(f"  CITL         est={est_citl:.3f}  day [{citl_day[0]:.3f},{citl_day[1]:.3f}]  cluster [{citl_cluster[0]:.3f},{citl_cluster[1]:.3f}]")
        print(f"  Brier-skill  est={est_brier_skill:+.3f}  day [{bs_day[0]:+.3f},{bs_day[1]:+.3f}]  cluster [{bs_cluster[0]:+.3f},{bs_cluster[1]:+.3f}]")
        print()

        rows.append({
            "target": tgt, "architecture": label,
            "leaf": str(leaf.relative_to(EXP)),
            "n_rows": int(len(y)), "n_patients": int(len(set(pid))),
            "auroc": round(est_auroc, 4),
            "auroc_day_lo": round(auroc_day[0], 4), "auroc_day_hi": round(auroc_day[1], 4),
            "auroc_cluster_lo": round(auroc_cluster[0], 4), "auroc_cluster_hi": round(auroc_cluster[1], 4),
            "slope": round(est_slope, 3),
            "slope_day_lo": round(slope_day[0], 3), "slope_day_hi": round(slope_day[1], 3),
            "slope_cluster_lo": round(slope_cluster[0], 3), "slope_cluster_hi": round(slope_cluster[1], 3),
            "citl": round(est_citl, 3),
            "citl_day_lo": round(citl_day[0], 3), "citl_day_hi": round(citl_day[1], 3),
            "citl_cluster_lo": round(citl_cluster[0], 3), "citl_cluster_hi": round(citl_cluster[1], 3),
            "brier_skill": round(est_brier_skill, 4),
            "bs_day_lo": round(bs_day[0], 4), "bs_day_hi": round(bs_day[1], 4),
            "bs_cluster_lo": round(bs_cluster[0], 4), "bs_cluster_hi": round(bs_cluster[1], 4),
        })

    if rows:
        df = pd.DataFrame(rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = HERE / f"patient_cluster_bootstrap_{ts}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
