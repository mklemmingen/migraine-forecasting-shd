"""Within-person C-statistic disaggregated by site / sex / base-rate stratum.

Computes the within-person C-statistic separately for each of the three
demographic / cohort strata:

- site: Uijeongbu vs Dongtan (the leave-one-site-out frame from the Results section)
- sex: female vs male (the demographic-narrowness frame from the Methods and Discussion sections)
- per-patient base-rate stratum: above vs below the cohort base rate
  (migraine 7.2%, headache 25%)

The point is to surface whether the Results section's pooled within-person C ≈ 0.55 result
masks within-group degradation - if (say) the male stratum gives a noticeably
lower or higher C, the pooled number hides equity-relevant variation that the
panel agent argued is load-bearing for any "this is a general population
finding" claim.

Per the panel ask, strata too small to estimate (e.g. male migraine where the
underlying cohort is 11 male patients and ~19 are estimable for migraine in
total, so the male migraine stratum may end up with 3-4 patients) are reported
quantitatively as "n_estimable = X (below the 5-patient reporting floor)"
rather than collapsed silently into the pooled view.

Uses the same CV-OOF predict path as fig_c5_within_person_calibration.py so
the estimable-patient count matches the Results section's 57 / 19 framework.

Usage: ``python experiment/_eval/_special/run_within_person_stratum.py``
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
EXP = HERE.parents[1]                                # experiment/
REPO = EXP.parent

sys.path[0:0] = [str(EXP), str(EXP / "5" / "_personal")]
from within_person import MIN_POS, per_patient_scores, within_person_cstatistic  # noqa: E402

WORKER = EXP / "5" / "_personal" / "_cv_oof_worker.py"
COHORT = REPO / "data" / "processed" / "special" / "cohort_metadata.parquet"

# Reporting floor: minimum estimable patients per stratum below which we refuse
# to compute a within-person C and instead report the count honestly.
STRATUM_REPORTING_FLOOR = 5

# Per-target cohort base rate (from the Results section) used to split patients into
# above/below base-rate strata.
COHORT_BASE_RATE = {"migraine": 0.072, "headache": 0.25}


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """CV-OOF predict; same pattern as fig_c5."""
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"wps_{uuid.uuid4().hex}.npz"
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
        return (
            z["y"].astype(int),
            z["p"].astype(float),
            z["pid"].astype(str),
        )
    finally:
        out.unlink(missing_ok=True)


def _resolve_headlines() -> dict[str, list[tuple[str, Path]]]:
    """Same composite-tracked leaf resolution as fig_c3/c4/c5."""
    spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    fp = mod.latest_figdata(EXP / "2")
    if fp is None:
        raise SystemExit("no figdata_*.json in experiment/2/; run experiment/2/compare.py first")
    print(f"  source {fp.name}")
    headlines = mod.load_figdata(fp).get("headlines", [])

    def find(target: str, family: str) -> Path | None:
        for role in ("headline", "runner_up"):
            for e in headlines:
                if (e.get("target") == target and e.get("family") == family
                        and e.get("feature_set") == "full_features"
                        and e.get("splittype") == "chrono"
                        and e.get("role") == role and e.get("leaf_dir")):
                    return Path(e["leaf_dir"])
        return None

    out: dict[str, list[tuple[str, Path]]] = {}
    for tgt in ("headache", "migraine"):
        leaves: list[tuple[str, Path]] = []
        for label, family in (("XGBoost stack", "xgboost"), ("TabPFN", "tabpfn")):
            d = find(tgt, family)
            if d is not None and (d / "model.joblib").exists():
                leaves.append((label, d))
        d4 = EXP / "4" / tgt / "full_features/sequence/version_window-mlp/70_15_15/chrono"
        if (d4 / "model.joblib").exists():
            leaves.append(("window-MLP", d4))
        out[tgt] = leaves
    return out


def _per_patient_base_rate(y: np.ndarray, pid: np.ndarray) -> pd.DataFrame:
    """Per-patient (n_days, n_pos, base_rate) table."""
    df = pd.DataFrame({"patient_id": pid, "y": y})
    return (df.groupby("patient_id")["y"]
            .agg(n_days="count", n_pos="sum")
            .reset_index()
            .assign(base_rate=lambda d: d["n_pos"] / d["n_days"]))


def _stratum_within_person_c(
    y: np.ndarray, p: np.ndarray, pid: np.ndarray,
    keep_pids: list[str], stratum_label: str,
) -> dict:
    """Compute the within-person C for the patients in ``keep_pids``. Returns
    a row dict with stratum label, n_estimable, the C-statistic + CI, and the
    "below floor" marker when the estimable count is too small to report."""
    keep_set = set(keep_pids)
    mask = np.isin(pid, list(keep_set))
    y_sub, p_sub, pid_sub = y[mask], p[mask], pid[mask]

    scores = per_patient_scores(y_sub, p_sub, pid_sub, MIN_POS)
    n_estimable = int(scores["estimable"].sum())
    if n_estimable < STRATUM_REPORTING_FLOOR:
        return {
            "stratum": stratum_label,
            "n_estimable": n_estimable,
            "below_floor": True,
            "within_c": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    res = within_person_cstatistic(scores)
    return {
        "stratum": stratum_label,
        "n_estimable": n_estimable,
        "below_floor": False,
        "within_c": round(float(res["estimate"]), 3),
        "ci_low": round(float(res["ci_low"]), 3),
        "ci_high": round(float(res["ci_high"]), 3),
    }


def main() -> None:
    cohort = pd.read_parquet(COHORT)
    print(f"cohort_metadata.parquet: {len(cohort)} patients "
          f"({(cohort['sex'] == 'female').sum()} female / "
          f"{(cohort['sex'] == 'male').sum()} male; "
          f"{(cohort['site'] == 'Uijeongbu').sum()} Uijeongbu / "
          f"{(cohort['site'] == 'Dongtan').sum()} Dongtan)")
    print()

    leaves = _resolve_headlines()
    rows: list[dict] = []

    for tgt, leaf_list in leaves.items():
        base = COHORT_BASE_RATE[tgt]
        print(f"=== {tgt} (cohort base rate {base:.1%}) ===")
        for label, leaf in leaf_list:
            t0 = time.time()
            print(f"  [{label}] predicting CV-OOF on {leaf.name}...")
            res = _predict(leaf)
            if res is None:
                continue
            y, p, pid = res
            print(f"  [{label}] done in {time.time()-t0:.0f}s  n_rows={len(y)}  pos_rate={y.mean():.3f}")

            br = _per_patient_base_rate(y, pid)
            br = br.merge(cohort, on="patient_id", how="left")
            br["br_stratum"] = np.where(br["base_rate"] >= base, "above", "below")

            # All-patient pooled (matches the Results section's reporting)
            all_pids = br["patient_id"].tolist()
            rows.append({
                "target": tgt, "architecture": label, "stratum_dim": "(pooled)",
                **_stratum_within_person_c(y, p, pid, all_pids, stratum_label="(pooled)"),
            })

            # Site stratum
            for site_val in ("Uijeongbu", "Dongtan"):
                sub_pids = br.loc[br["site"] == site_val, "patient_id"].tolist()
                rows.append({
                    "target": tgt, "architecture": label, "stratum_dim": "site",
                    **_stratum_within_person_c(y, p, pid, sub_pids,
                                                stratum_label=f"site={site_val}"),
                })

            # Sex stratum
            for sex_val in ("female", "male"):
                sub_pids = br.loc[br["sex"] == sex_val, "patient_id"].tolist()
                rows.append({
                    "target": tgt, "architecture": label, "stratum_dim": "sex",
                    **_stratum_within_person_c(y, p, pid, sub_pids,
                                                stratum_label=f"sex={sex_val}"),
                })

            # Base-rate stratum
            for br_val in ("above", "below"):
                sub_pids = br.loc[br["br_stratum"] == br_val, "patient_id"].tolist()
                rows.append({
                    "target": tgt, "architecture": label, "stratum_dim": "base_rate",
                    **_stratum_within_person_c(y, p, pid, sub_pids,
                                                stratum_label=f"base_rate {br_val} {base:.1%}"),
                })

            for r in rows[-9:]:
                if r["below_floor"]:
                    print(f"    {r['stratum']:<32} n={r['n_estimable']:>2}  (below {STRATUM_REPORTING_FLOOR}-patient reporting floor)")
                else:
                    print(f"    {r['stratum']:<32} n={r['n_estimable']:>2}  "
                          f"C={r['within_c']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
            print()

    df = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = HERE / f"within_person_stratum_{ts}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
