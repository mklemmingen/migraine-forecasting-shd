"""Hartung-Knapp-Sidik-Jonkman (HKSJ) t-interval sensitivity for the
within-person C-statistic at the canonical TabPFN headline cells, with
the Roever-Knapp-Friede 2015 modified variant (mKH) applied to the
small-k imbalanced-precision regime the migraine cell sits in.

The body §3.6 within-person C-statistic CIs are PM random-effects with
the normal-approximation 1.96 * SE interval. Roever et al. 2015 (BMC
Med Res Methodol 15:99, PMC4647507) recommend the modified mKH
procedure for ``few studies + imbalanced precisions``, the regime the
migraine k=19 cell occupies. The headache k=57 cell is in a less
constrained regime; reporting HKSJ alongside there provides a
side-by-side coverage check rather than a regime correction.

This runner produces both targets' HKSJ-corrected CIs alongside the PM
normal-approximation CIs already in
``within_person_pooling_comparison_<ts>.csv``, so body §3.6 can either
swap the migraine cell's primary CI to mKH or report both side-by-side.

Output: one summary CSV per run under
``experiment/_eval/_special/hksj_within_person_<ts>.csv``.

Usage: ``python experiment/_eval/_special/run_hksj_within_person.py``
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
from within_person import (  # noqa: E402
    MIN_POS,
    hksj_within_person_ci,
    per_patient_scores,
    within_person_cstatistic,
)

WORKER = EXP / "5" / "_personal" / "_cv_oof_worker.py"


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path):
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"hksj_{uuid.uuid4().hex}.npz"
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
        pm = within_person_cstatistic(scores, method="PM")
        hksj = hksj_within_person_ci(scores, alpha=0.05)

        pm_width = pm["ci_high"] - pm["ci_low"]
        hksj_width = hksj["ci_high"] - hksj["ci_low"]
        ratio = hksj_width / pm_width if pm_width > 0 else float('nan')

        print(f"  k_estimable = {pm['k_estimable']}  median_AUROC = {pm['median_auroc']:.4f}")
        print(f"  PM normal-approx 95% CI: C = {pm['estimate']:.4f} "
              f"[{pm['ci_low']:.4f}, {pm['ci_high']:.4f}]  width = {pm_width:.4f}")
        print(f"  HKSJ mKH t-interval:    C = {hksj['estimate']:.4f} "
              f"[{hksj['ci_low']:.4f}, {hksj['ci_high']:.4f}]  width = {hksj_width:.4f}")
        print(f"  HKSJ q = {hksj['q']:.4f}  q_mod = {hksj['q_mod']:.4f}  "
              f"df = {hksj['df']}  t_crit = {hksj['t_crit']:.4f}")
        print(f"  width ratio (HKSJ / PM) = {ratio:.3f}")
        print()

        rows.append({
            "target": tgt, "architecture": "TabPFN",
            "leaf": str(leaf.relative_to(EXP)),
            "k_estimable": pm["k_estimable"],
            "median_auroc": round(pm["median_auroc"], 4),
            "pm_estimate": round(pm["estimate"], 4),
            "pm_ci_low": round(pm["ci_low"], 4),
            "pm_ci_high": round(pm["ci_high"], 4),
            "pm_ci_width": round(pm_width, 4),
            "hksj_estimate": round(hksj["estimate"], 4),
            "hksj_ci_low": round(hksj["ci_low"], 4),
            "hksj_ci_high": round(hksj["ci_high"], 4),
            "hksj_ci_width": round(hksj_width, 4),
            "hksj_q_raw": round(hksj["q"], 4),
            "hksj_q_mod": round(hksj["q_mod"], 4),
            "hksj_df": hksj["df"],
            "hksj_t_crit": round(hksj["t_crit"], 4),
            "width_ratio_hksj_over_pm": round(ratio, 3),
        })

    if rows:
        df = pd.DataFrame(rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = HERE / f"hksj_within_person_{ts}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv.relative_to(REPO)}")


if __name__ == "__main__":
    main()
