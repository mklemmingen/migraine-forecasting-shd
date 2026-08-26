"""Pooled within-person C-statistic computed under both Paule-Mandel (the
current primary) and DerSimonian-Laird (legacy comparability sensitivity)
for the canonical TabPFN headline cells.

DerSimonian-Laird underestimates τ² at k < 20 per Veroniki 2016, which is
the regime this cohort sits in (k=19 estimable patients on migraine,
k=57 on headache); Paule-Mandel τ² is robust in that regime. This script
delivers the comparison values used in the Results section and abstract: PM as primary,
DL retained as side-by-side sensitivity so the literature-comparability
baseline is preserved.

Uses the same CV-OOF prediction path as fig_c5 + the within-person stratum
runner so the per-patient AUROC distribution is the same one the Results
section cites; only the pooling step changes between DL and PM. Output CSV is
timestamped to preserve prior runs per the never-overwrite convention LOSO
summary CSVs established.

Usage: ``python experiment/_eval/_special/run_within_person_pooling_comparison.py``
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

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
REPO = EXP.parent

sys.path[0:0] = [str(EXP), str(EXP / "5" / "_personal")]
from within_person import MIN_POS, per_patient_scores, within_person_cstatistic  # noqa: E402

WORKER = EXP / "5" / "_personal" / "_cv_oof_worker.py"


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path):
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"wpc_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run(
            [sys.executable, str(WORKER), str(leaf), str(out)],
            capture_output=True, text=True, timeout=3600, env=_env(addition),
        )
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr.strip().splitlines() or ["worker failed"])[-1]
            print(f"  SKIP {leaf.name}: {tail}")
            return None
        z = np.load(out, allow_pickle=True)
        return (z["y"].astype(int), z["p"].astype(float), z["pid"].astype(str))
    finally:
        out.unlink(missing_ok=True)


def _resolve_tabpfn_headlines() -> dict[str, Path]:
    """Resolve TabPFN headline cells for both targets (the architecture cited
    in the Results section). Mirrors fig_c5 / within-person stratum runner."""
    spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    fp = mod.latest_figdata(EXP / "2")
    if fp is None:
        raise SystemExit("no figdata_*.json in experiment/2/")
    print(f"  source {fp.name}")
    headlines = mod.load_figdata(fp).get("headlines", [])
    out: dict[str, Path] = {}
    for tgt in ("headache", "migraine"):
        for role in ("headline", "runner_up"):
            for e in headlines:
                if (e.get("target") == tgt and e.get("family") == "tabpfn"
                        and e.get("feature_set") == "full_features"
                        and e.get("splittype") == "chrono"
                        and e.get("role") == role and e.get("leaf_dir")):
                    out[tgt] = Path(e["leaf_dir"])
                    break
            if tgt in out:
                break
        if tgt not in out:
            raise SystemExit(f"no composite-tracked TabPFN leaf for {tgt}/full_features/chrono")
    return out


def main() -> None:
    leaves = _resolve_tabpfn_headlines()
    for tgt, leaf in leaves.items():
        print(f"  {tgt:<9} TabPFN  -> {leaf.relative_to(EXP)}")
    print()

    rows: list[dict] = []
    for tgt, leaf in leaves.items():
        print(f"=== {tgt} (TabPFN headline cell) ===")
        t0 = time.time()
        r = _predict(leaf)
        if r is None:
            continue
        y, p, pid = r
        print(f"  CV-OOF done in {time.time()-t0:.0f}s; n_rows={len(y)}, pos_rate={y.mean():.3f}")

        scores = per_patient_scores(y, p, pid, MIN_POS)
        # within_person_cstatistic now returns both PM (primary) and DL (sensitivity).
        res = within_person_cstatistic(scores, method="PM")
        print(f"  PM (primary): C = {res['estimate']:.3f} "
              f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  τ² = {res['tau2']:.5f}  "
              f"k_estimable = {res['k_estimable']}")
        print(f"  DL (sensit):  C = {res['estimate_dl']:.3f} "
              f"[{res['ci_low_dl']:.3f}, {res['ci_high_dl']:.3f}]  τ² = {res['tau2_dl']:.5f}")
        print()

        rows.append({
            "target": tgt, "architecture": "TabPFN",
            "leaf": str(leaf.relative_to(EXP)),
            "k_estimable": res["k_estimable"],
            "median_auroc": round(res["median_auroc"], 4),
            "pm_estimate": round(res["estimate_pm"], 4),
            "pm_ci_low": round(res["ci_low_pm"], 4),
            "pm_ci_high": round(res["ci_high_pm"], 4),
            "pm_tau2": round(res["tau2_pm"], 6),
            "dl_estimate": round(res["estimate_dl"], 4),
            "dl_ci_low": round(res["ci_low_dl"], 4),
            "dl_ci_high": round(res["ci_high_dl"], 4),
            "dl_tau2": round(res["tau2_dl"], 6),
        })

    if rows:
        df = pd.DataFrame(rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = HERE / f"within_person_pooling_comparison_{ts}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
