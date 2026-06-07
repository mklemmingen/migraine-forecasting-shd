"""Medication-overuse-headache (MOH) feasibility check on headline-cell
predicted probabilities. Computes the predicted-positive fraction
``P(yhat > t)`` at decision-curve thresholds t=0.05 and t=0.10 on the
canonical migraine and headache headline test sets, both pooled and
per-patient. Compares against the ICHD-3 2018 8.2 MOH ceilings:

- triptan-overuse headache (8.2.2): ``>= 10 days/month`` triptan intake
  (33.3% of a 30-day month)
- NSAID-overuse headache (8.2.3.2): ``>= 15 days/month`` NSAID intake
  (50.0% of a 30-day month)

The body §3.7 decision-curve paragraph invokes the MOH ceiling as an
implicit upper limit on trigger rate but does not operationalise whether
the cited +0.152 headache net benefit at t=0.05 would be MOH-feasible
given the 7.2% pooled positive rate. This runner closes that loop with
explicit per-patient trigger-rate distributions.

The Park 2016 cohort by inclusion criteria carries 2-14 headache days
per month (episodic band, well below chronic-migraine 1.3 >= 15-day
threshold), so the MOH framing applies as a pre-emptive abortive
consumption ceiling rather than a chronic-MOH diagnostic ceiling per
the ICHD-3 2018 verbatim quoted in the body §2.6 paragraph.

Output: one summary CSV per run under
``experiment/_eval/_special/moh_trigger_rate_<ts>.csv``.

Usage: ``python experiment/_eval/_special/run_moh_trigger_rate.py``
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

sys.path.insert(0, str(EXP))

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"

TRIPTAN_CEILING = 10.0 / 30.0
NSAID_CEILING = 15.0 / 30.0
THRESHOLDS = (0.05, 0.10)


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path):
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"moh_{uuid.uuid4().hex}.npz"
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


def _resolve_headline_leaves() -> dict[str, Path]:
    spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    fp = mod.latest_figdata(EXP / "2")
    if fp is None:
        raise SystemExit("no figdata_*.json in experiment/2/; run experiment/2/compare.py first")
    print(f"resolving headlines from {fp.name}")
    headlines = mod.load_figdata(fp).get("headlines", [])
    family_for_target = {"migraine": "xgboost", "headache": "tabpfn"}
    out: dict[str, Path] = {}
    for tgt in ("migraine", "headache"):
        family = family_for_target[tgt]
        leaf = None
        for role in ("headline", "runner_up"):
            for e in headlines:
                if (e.get("target") == tgt and e.get("family") == family
                        and e.get("feature_set") == "full_features"
                        and e.get("splittype") == "chrono"
                        and e.get("role") == role and e.get("leaf_dir")):
                    leaf = Path(e["leaf_dir"])
                    break
            if leaf is not None:
                break
        if leaf is None:
            raise SystemExit(f"no composite-tracked headline leaf for {tgt}/{family}/full_features/chrono")
        out[tgt] = leaf
        print(f"  {tgt:<9} {family:<8} -> {leaf.relative_to(EXP)}")
    return out


def _trigger_rates(y: np.ndarray, p: np.ndarray, pid: np.ndarray, t: float) -> dict:
    """Pooled and per-patient (yhat > t) rates with summary statistics."""
    flags = (p > t).astype(int)
    pooled = float(flags.mean())
    df = pd.DataFrame({"pid": pid, "flag": flags})
    per_patient = df.groupby("pid")["flag"].mean().to_numpy(dtype=float)
    n_patients_over_triptan = int((per_patient > TRIPTAN_CEILING).sum())
    n_patients_over_nsaid = int((per_patient > NSAID_CEILING).sum())
    return {
        "pooled_trigger_rate": pooled,
        "per_patient_median": float(np.median(per_patient)),
        "per_patient_q25": float(np.percentile(per_patient, 25)),
        "per_patient_q75": float(np.percentile(per_patient, 75)),
        "per_patient_max": float(per_patient.max()),
        "n_patients_over_triptan_ceiling": n_patients_over_triptan,
        "n_patients_over_nsaid_ceiling": n_patients_over_nsaid,
        "n_patients_total": int(len(per_patient)),
    }


def main() -> None:
    print(f"ICHD-3 2018 8.2 MOH ceilings:")
    print(f"  triptan (8.2.2): >=10 days/month  =  {TRIPTAN_CEILING:.4f} fraction")
    print(f"  NSAID   (8.2.3.2): >=15 days/month =  {NSAID_CEILING:.4f} fraction")
    print()

    leaves = _resolve_headline_leaves()
    print()

    rows: list[dict] = []
    for tgt, leaf in leaves.items():
        print(f"=== {tgt} headline leaf: {leaf.name} ===")
        result = _predict(leaf)
        if result is None:
            continue
        y, p, pid = result
        print(f"  predicted: n_rows={len(y)}, n_pos={y.sum()}, "
              f"pos_rate={y.mean():.4f}, n_patients={len(np.unique(pid))}")

        for t in THRESHOLDS:
            stats = _trigger_rates(y, p, pid, t)
            print(f"  t={t:.2f}:  pooled={stats['pooled_trigger_rate']:.3f}  "
                  f"per_patient median={stats['per_patient_median']:.3f} "
                  f"[Q1={stats['per_patient_q25']:.3f}, Q3={stats['per_patient_q75']:.3f}]  "
                  f"max={stats['per_patient_max']:.3f}")
            print(f"           patients_over_triptan_ceiling="
                  f"{stats['n_patients_over_triptan_ceiling']}/{stats['n_patients_total']}  "
                  f"patients_over_nsaid_ceiling="
                  f"{stats['n_patients_over_nsaid_ceiling']}/{stats['n_patients_total']}")
            rows.append({
                "target": tgt, "leaf": str(leaf.relative_to(EXP)),
                "threshold": t, **stats,
                "triptan_ceiling_fraction": TRIPTAN_CEILING,
                "nsaid_ceiling_fraction": NSAID_CEILING,
            })
        print()

    if rows:
        df = pd.DataFrame(rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = HERE / f"moh_trigger_rate_{ts}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
