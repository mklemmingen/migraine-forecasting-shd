"""Build figures/headline_predictions.npz, the prediction cache fig_m4 draws from.

The decision-curve panel of `fig_m_evaluation_series.py` needs per-day predictions,
not summary statistics: net benefit is recomputed at every threshold from y and p.
The result CSVs only carry endpoints, so the predictions are cached here once.

The cache is small (~8 KB) and tracked, matching how every other derived result in
this repo is handled, because regenerating it runs the Addition-5 predict worker over
both headline leaves and takes roughly a quarter of an hour on CPU.

Faithfulness gate: migraine XGB-HP020 reproduces its published AUROC exactly, and the
script asserts that. The headache TabPFN cell does NOT reproduce on a machine without
a GPU -- CPU inference of that cell shifts its AUROC by about 0.016 -- so its
predictions are cached but deliberately not drawn as a curve (see TODO H10). Rerun
this on the ROCm box to get a headache cache that may be drawn.

Usage: python make_headline_predictions.py
"""
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiment"
WORKER = EXP / "5" / "_personal" / "_predict_worker.py"
OUT = HERE / "figures" / "headline_predictions.npz"

# the two headline cells, as named in Results
LEAVES = {
    "migraine": EXP / "0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono"
                      "/HyperparameterTuned/single_AUROC/HP020",
    "headache": EXP / "1/headache/full_features/tabpfn/version_2-6/70_30/chrono",
}
PUBLISHED = {"migraine": 0.7913, "headache": 0.6530}


def main() -> None:
    store = {}
    for target, leaf in LEAVES.items():
        if not leaf.exists():
            raise SystemExit(f"leaf missing: {leaf}")
        tmp = Path(tempfile.gettempdir()) / f"hp_{uuid.uuid4().hex}.npz"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        try:
            r = subprocess.run([sys.executable, str(WORKER), str(leaf), str(tmp)],
                               capture_output=True, text=True, timeout=3600, env=env)
            if r.returncode != 0 or not tmp.exists():
                tail = r.stderr.strip().splitlines()[-3:] if r.stderr else []
                raise SystemExit(f"{target}: predict worker failed\n" + "\n".join(tail))
            z = np.load(tmp)
            store[f"{target}_y"] = z["y"].astype(float)
            store[f"{target}_p"] = z["p"].astype(float)
            store[f"{target}_pid"] = z["pid"]
        finally:
            tmp.unlink(missing_ok=True)
        auc = float(roc_auc_score(store[f"{target}_y"], store[f"{target}_p"]))
        delta = auc - PUBLISHED[target]
        print(f"  {target}: n={len(store[f'{target}_y'])} AUROC={auc:.4f} "
              f"published={PUBLISHED[target]:.4f} delta={delta:+.4f}", flush=True)

    # only migraine is drawn as a curve, so only migraine has to be exact
    mig = float(roc_auc_score(store["migraine_y"], store["migraine_p"]))
    assert abs(mig - PUBLISHED["migraine"]) < 5e-4, \
        f"migraine predictions do not reproduce the published AUROC ({mig:.4f})"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **store)
    print(f"saved {OUT.relative_to(HERE)}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
