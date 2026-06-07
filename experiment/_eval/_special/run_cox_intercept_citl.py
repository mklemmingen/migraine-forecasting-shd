"""Cox calibration-in-the-large (CITL) alpha with slope fixed at 1, computed
on the canonical migraine and headache headline cells (Huang et al. 2020,
JAMIA 27(4):621-633, equation 7: ``logit{P(O=1)} = alpha + beta * logit(E)``
with beta constrained to 1).

The body §3.4 calibration paragraph reports the O:E ratio (mean(y) / mean(p))
under the name "CITL", but Huang 2020 specifies CITL as the Cox-form
alpha with the slope constrained at 1. The two quantities are distinct:
the O:E ratio is the ratio of mean outcome to mean prediction on the
probability scale, while the Cox alpha is the logit-scale shift required
to recover mean calibration with the model's discrimination intact.
This runner produces the Cox alpha + patient-cluster bootstrap 95% CI
so the body can either rename O:E or report Cox alpha alongside.

Uses the same predict-worker path as run_no_aura_sensitivity so the
headline-cell test-set predictions are the canonical composite-tracked
ones the §3.2 numbers cite. Output: one summary CSV per run under
``experiment/_eval/_special/cox_intercept_citl_<ts>.csv``.

Usage: ``python experiment/_eval/_special/run_cox_intercept_citl.py``
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
from _eval.metrics_lib import (  # noqa: E402
    calibration_intercept,
    calibration_slope,
    cox_intercept_citl,
)

WORKER = EXP / "5" / "_personal" / "_predict_worker.py"


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path):
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"cox_{uuid.uuid4().hex}.npz"
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


def _patient_cluster_bootstrap_cox(y: np.ndarray, p: np.ndarray, pid: np.ndarray,
                                   n_iter: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Patient-cluster bootstrap of the Cox CITL alpha. Returns
    (point_estimate, ci_low, ci_high) at 95% percentile interval."""
    rng = np.random.default_rng(seed)
    patients = np.unique(pid)
    point = cox_intercept_citl(y, p)
    samples: list[float] = []
    for _ in range(n_iter):
        draw = rng.choice(patients, size=len(patients), replace=True)
        rows = np.concatenate([np.where(pid == d)[0] for d in draw])
        alpha = cox_intercept_citl(y[rows], p[rows])
        if np.isfinite(alpha):
            samples.append(alpha)
    if not samples:
        return point, float('nan'), float('nan')
    arr = np.asarray(samples, dtype=float)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    leaves = _resolve_headline_leaves()
    print()

    rows: list[dict] = []
    for tgt, leaf in leaves.items():
        print(f"=== {tgt} headline leaf: {leaf.name} ===")
        result = _predict(leaf)
        if result is None:
            continue
        y, p, pid = result
        print(f"  predicted: n_rows={len(y)}, n_pos={y.sum()}, pos_rate={y.mean():.4f}")

        cox_point, cox_lo, cox_hi = _patient_cluster_bootstrap_cox(y, p, pid)
        platt_alpha = calibration_intercept(y, p)
        slope = calibration_slope(y, p)
        oe = float(y.mean() / p.mean()) if p.mean() > 0 else float('nan')

        print(f"  Cox CITL alpha (beta fixed at 1):  {cox_point:+.4f} "
              f"[{cox_lo:+.4f}, {cox_hi:+.4f}]   (Huang 2020 eq.7)")
        print(f"  Platt-style intercept (joint fit): {platt_alpha:+.4f}   "
              f"slope={slope:.3f}")
        print(f"  Observed/Expected ratio:           {oe:.4f}   "
              f"(currently labelled 'CITL' in body §3.4)")
        print()

        rows.append({
            "target": tgt, "leaf": str(leaf.relative_to(EXP)),
            "n_rows": int(len(y)), "n_pos": int(y.sum()),
            "pos_rate": round(float(y.mean()), 4),
            "cox_citl_alpha": round(cox_point, 4),
            "cox_citl_ci_low": round(cox_lo, 4),
            "cox_citl_ci_high": round(cox_hi, 4),
            "platt_intercept": round(platt_alpha, 4),
            "platt_slope": round(slope, 3),
            "oe_ratio": round(oe, 4),
        })

    if rows:
        df = pd.DataFrame(rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = HERE / f"cox_intercept_citl_{ts}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
