"""Export per-patient AUROCs from the cached out-of-fold predictions.

`fig_c2_within_person.py` caches each headline leaf's out-of-fold predictions
under `figures/_cv_oof_cache/`. This script turns that cache into a plain CSV so
downstream consumers (notably `graphical_abstract.py`) can plot the per-patient
distribution without re-running the CV refit, which costs minutes per leaf on
CPU.

Estimability and the C-statistic come from `experiment/5/_personal/within_person.py`
-- the same estimator the figure and the manuscript use -- rather than being
recomputed here, so the CSV cannot drift from the published values.

The script refuses to write if the recomputed within-person C does not match the
value published for that leaf, so a stale or corrupted cache fails loudly instead
of silently producing a wrong figure.

Usage: python export_per_patient_auroc.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path.insert(0, str(EXP / "5" / "_personal"))
sys.path.insert(0, str(EXP))
import within_person as WP  # noqa: E402

CACHE = HERE / "figures" / "_cv_oof_cache"
OUT = HERE / "figures" / "per_patient_auroc.csv"

# Published within-person C per target (within_person_pooling_comparison CSV and
# article.tex Table 2). The export is gated on reproducing these.
# Gate values are per (target, architecture): the graphical abstract's hero shows the
# migraine XGB stack (within-person C 0.5576, Table 2 rounds to 0.558), NOT the
# TabPFN-v2.5-finetuned cell (0.5653) that Supplementary Figure S2 panel b uses. The
# per-patient panel must come from the same cell as the hero or the two disagree.
PUBLISHED = {"headache": (0.5418, 57), "migraine": (0.5576, 19)}
TOL = 0.002


def main() -> None:
    if not CACHE.is_dir() or not any(CACHE.glob("*.npz")):
        raise SystemExit(f"no cache in {CACHE} - run fig_c2_within_person.py first")
    rows, ok = [], True
    for npz in sorted(CACHE.glob("*.npz")):
        target = "headache" if "__headache__" in npz.name or "/headache/" in npz.name \
            else ("migraine" if "migraine" in npz.name else None)
        if target is None:
            print(f"  skip {npz.name}: cannot infer target")
            continue
        z = np.load(npz)
        y, p, pid = z["y"].astype(float), z["p"].astype(float), z["pid"].astype(str)
        scores = WP.per_patient_scores(y, p, pid, WP.MIN_POS)
        est = scores[scores["estimable"]].sort_values("auroc").reset_index(drop=True)
        within = WP.within_person_cstatistic(scores)
        exp_c, exp_k = PUBLISHED[target]
        match = abs(within["estimate"] - exp_c) <= TOL and len(est) == exp_k
        print(f"  {target:<9} within-person C {within['estimate']:.4f} (published {exp_c}) "
              f"| k={len(est)} (published {exp_k}) -> {'MATCH' if match else 'MISMATCH'}")
        if not match:
            ok = False
        for i, r in enumerate(est.itertuples()):
            rows.append({"target": target, "rank": i, "patient": r.patient,
                         "auroc": float(r.auroc), "n": int(r.n), "n_pos": int(r.n_pos)})
    if not ok:
        raise SystemExit("refusing to write: recomputed values do not match the published ones")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"saved {OUT.name}  ({len(df)} patient rows)")


if __name__ == "__main__":
    main()
