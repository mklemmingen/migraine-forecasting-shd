"""Run the n=2 with-aura sensitivity analysis on the canonical headline cells.

Loads the patient-level aura flags persisted by ``data/pipeline/_special/aura.py``
at ``data/processed/special/aura_status.parquet``, resolves the migraine and
headache headline leaves from the latest ``experiment/2/figdata_*.json`` (so the
composite-tracked headline cells the body §3.2 cites are the ones evaluated),
invokes the existing ``experiment/5/_personal/_predict_worker.py`` to regenerate
the val+test (y, p, patient_id) arrays per leaf, and recomputes the headline
discrimination + calibration metrics on the no-aura subset.

The runner reports the shift between the full-cohort headline numbers and the
no-aura-subset numbers, demonstrating that the n=2 aura subset's retention in
the headline analyses does not drive the cited findings. Output: one summary
CSV per run under ``experiment/_eval/_special/no_aura_sensitivity_<ts>.csv``.

Patients with unknown aura status (CMC-0066, the one body §3.1 deduplication
split-daughter without a row in the raw baseline sheet) are treated as no-aura
for the sensitivity: the 2 aura patients are uniquely identified at the raw
level (CMC-0026 and DHA-0057), and any deduplication-created daughter inherits
the parent's no-aura status because neither aura patient has a sibling ID.

Usage: ``python experiment/_eval/_special/run_no_aura_sensitivity.py``
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]                                # experiment/
REPO = EXP.parent

sys.path.insert(0, str(EXP))
from _eval.metrics_lib import calibration_intercept, calibration_slope  # noqa: E402

AURA_STATUS = REPO / "data" / "processed" / "special" / "aura_status.parquet"
WORKER = EXP / "5" / "_personal" / "_predict_worker.py"


def _env(addition: str) -> dict:
    """Mirror the env policy of run_external_site._env: keep the GPU visible only
    for the TabPFN Addition-1 worker."""
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Run the existing predict worker on this leaf and return (y, p, patient_id).
    Returns None if the worker fails."""
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"nas_{uuid.uuid4().hex}.npz"
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
        return (
            z["y"].astype(int),
            z["p"].astype(float),
            z["pid"].astype(str),
        )
    finally:
        out.unlink(missing_ok=True)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """AUROC, calibration slope and intercept, Brier score, and Brier skill
    against the cohort base rate (climatology = mean(y)). Returns NaN where
    a metric is undegenerate (e.g. AUROC needs both classes present)."""
    out: dict = {
        "n_rows": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": round(float(y.mean()), 4) if len(y) else float("nan"),
    }
    if len(y) and len(set(y)) > 1:
        out["auroc"] = round(float(roc_auc_score(y, p)), 4)
    else:
        out["auroc"] = float("nan")
    out["brier"] = round(float(brier_score_loss(y, p)), 5) if len(y) else float("nan")
    if len(y):
        clim = float(y.mean())
        clim_brier = clim * (1 - clim)
        out["brier_skill_vs_pooled_climatology"] = (
            round(1.0 - out["brier"] / clim_brier, 4) if clim_brier > 0 else float("nan")
        )
    out["cal_slope"] = round(calibration_slope(y, p), 3)
    out["cal_intercept_alpha"] = round(calibration_intercept(y, p), 3)
    return out


def _resolve_headline_leaves() -> dict[str, Path]:
    """Resolve the migraine and headache headline leaves from the latest
    experiment/2/figdata_*.json, following the same composite-tracking
    discipline as run_external_site._load_figdata_headlines."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
    mod = ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    fp = mod.latest_figdata(EXP / "2")
    if fp is None:
        raise SystemExit("no figdata_*.json in experiment/2/; run experiment/2/compare.py first")
    print(f"resolving headlines from {fp.name}")
    headlines = mod.load_figdata(fp).get("headlines", [])

    out: dict[str, Path] = {}
    for tgt in ("migraine", "headache"):
        # The headline cell per body §3.2: migraine XGB-HP020, headache TabPFN-v2.6.
        # Headline composite picks among "headline" roles first; if the composite
        # rule landed the family elsewhere we still want the named headline cell
        # so we search across both roles per (target, family) ordering.
        family_for_target = {"migraine": "xgboost", "headache": "tabpfn"}
        family = family_for_target[tgt]
        leaf = None
        for role in ("headline", "runner_up"):
            for e in headlines:
                if (e.get("target") == tgt and e.get("family") == family
                        and e.get("feature_set") == "full_features"
                        and e.get("splittype") == "chrono"
                        and e.get("role") == role
                        and e.get("leaf_dir")):
                    leaf = Path(e["leaf_dir"])
                    break
            if leaf is not None:
                break
        if leaf is None:
            raise SystemExit(f"no composite-tracked headline leaf for {tgt}/{family}/full_features/chrono")
        out[tgt] = leaf
        print(f"  {tgt:<9} {family:<8} -> {leaf.relative_to(EXP)}")
    return out


def main() -> None:
    aura = pd.read_parquet(AURA_STATUS)
    aura_pids = set(aura.loc[aura["has_aura"], "patient_id"])
    print(f"aura_status.parquet: {len(aura)} patients, {len(aura_pids)} with aura ({sorted(aura_pids)})")
    print()

    leaves = _resolve_headline_leaves()
    print()

    rows: list[dict] = []
    for tgt, leaf in leaves.items():
        print(f"=== {tgt} headline leaf: {leaf.name} ===")
        result = _predict(leaf)
        if result is None:
            print(f"  SKIP (predict worker failed)")
            continue
        y, p, pid = result
        # patient_id from the predict worker is the canonical processed format
        # (uppercased CMC-XXXX / DHA-XXXX); the aura set is also uppercased so
        # set membership is a direct string compare.
        no_aura_mask = ~np.isin(pid, list(aura_pids))

        full = _metrics(y, p)
        sub = _metrics(y[no_aura_mask], p[no_aura_mask])

        n_excluded = int(len(y) - no_aura_mask.sum())
        excluded_pids = sorted(set(pid[~no_aura_mask].tolist())) if n_excluded else []
        print(f"  full cohort:  n_rows={full['n_rows']:>4}  AUROC={full['auroc']:.4f}  slope={full['cal_slope']:>5}  Brier={full['brier']:.5f}")
        print(f"  no-aura set:  n_rows={sub['n_rows']:>4}  AUROC={sub['auroc']:.4f}  slope={sub['cal_slope']:>5}  Brier={sub['brier']:.5f}")
        print(f"  excluded:     n_rows={n_excluded:>4} from {len(excluded_pids)} aura patient(s) {excluded_pids}")
        print(f"  delta:        ΔAUROC={sub['auroc'] - full['auroc']:+.4f}  Δslope={sub['cal_slope'] - full['cal_slope']:+.3f}  ΔBrier={sub['brier'] - full['brier']:+.5f}")
        print()

        for label, m, n_excl, pids in (("full_cohort", full, 0, []),
                                        ("no_aura", sub, n_excluded, excluded_pids)):
            row = {"target": tgt, "leaf": str(leaf.relative_to(EXP)),
                   "subset": label, **m,
                   "n_aura_excluded_rows": n_excl,
                   "aura_excluded_pids": ",".join(pids)}
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = HERE / f"no_aura_sensitivity_{ts}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
