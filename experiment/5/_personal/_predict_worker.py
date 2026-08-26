"""Subprocess prediction worker for the Addition 5 within-person evaluation.

Run in a FRESH interpreter per leaf so the two conflicting top-level
``_model_architecture`` packages (experiment/0 for the XGBoost calibrated
closure, experiment/4 for the sequence estimator's nn.Modules) and the differing
devices (TabPFN on the ROCm GPU, XGBoost/sequence on CPU) never collide in one
process. Each addition gets exactly the sys.path it needs.

Regenerates predictions on the model's full out-of-sample horizon (val + test),
retaining patient_id for within-person grouping.

Usage: python _predict_worker.py <leaf_model_dir> <out_npz>
"""
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]   # experiment/
REPO = EXP.parent
RATIOS = ("70_15_15", "70_30", "80_20")
SPLITS = ("chrono", "stratified", "patient")


def _dims(model_dir: Path) -> dict:
    parts = model_dir.relative_to(EXP).parts
    return {"addition": parts[0], "target": parts[1], "feature_set": parts[2],
            "ratio": next((p for p in parts if p in RATIOS), "?"),
            "split": next((p for p in parts if p in SPLITS), "?"),
            "is_seq": "sequence" in parts}


def _force_cpu_unpickling() -> None:
    """Let bundles fitted on a CUDA box load on a CPU-only machine.

    The TabPFN bundles carry CUDA-backed tensors. joblib unpickles them through
    torch's storage loader, which calls torch.load itself and so ignores any
    map_location the caller sets -- the load dies with "Attempting to
    deserialize object on a CUDA device". Redirecting _load_from_bytes is the
    documented way to force that nested load onto the CPU. weights_only stays
    False because these are full estimator pickles, not bare state dicts.
    """
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        return
    import io as _io
    torch.storage._load_from_bytes = lambda b: torch.load(
        _io.BytesIO(b), map_location="cpu", weights_only=False)


def _to_cpu(bundle):
    """Move a bundle fitted on a GPU box onto the CPU before predicting.

    Clearing the estimator's own device attributes is not enough: the fitted
    TabPFN inference engine keeps its own model cache keyed by torch.device, and
    the memory heuristic reads the devices from that cache, so it still takes the
    CUDA branch and dies on a CPU-only build. The cache exposes .to(devices),
    which moves the weights and rekeys in one step.
    """
    try:
        import torch
    except ImportError:
        return bundle
    if torch.cuda.is_available():
        return bundle
    cpu = torch.device("cpu")
    if getattr(bundle, "device", None) not in (None, "cpu"):
        bundle.device = "cpu"
    if hasattr(bundle, "devices_"):
        bundle.devices_ = (cpu,)
    for cache in getattr(getattr(bundle, "executor_", None), "model_caches", []) or []:
        cache.to([cpu])
    return bundle


def main(model_dir: str, out: str) -> None:
    import joblib
    import pandas as pd

    _force_cpu_unpickling()

    md = Path(model_dir).resolve()
    d = _dims(md)
    sys.path.insert(0, str(EXP))
    from _dataRead.read import load_raw, prep_split  # noqa: E402
    from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
    from _dataRead.filter_to_park_features import select_park_features  # noqa: E402
    loaders = {"full_features": None,
               "no_rolling_features": select_non_rolling_features,
               "park_features": select_park_features}

    base = REPO / "data" / "processed" / d["target"] / d["ratio"] / d["split"]
    loader = loaders.get(d["feature_set"])
    frames = [load_raw(str(base / f"diary_{sp}.parquet"), loader=loader)
              for sp in ("val", "test") if (base / f"diary_{sp}.parquet").exists()]
    df = pd.concat(frames, ignore_index=True)
    pid = df["patient_id"].to_numpy().astype(str)

    if d["is_seq"]:
        # Sequence estimator: needs experiment/4 (+ _seq) on the path so joblib
        # can unpickle SequenceClassifier (sklearn_wrapper, windowing) and the
        # nn.Modules under experiment/4/_model_architecture.
        sys.path[0:0] = [str(EXP / "4"), str(EXP / "4" / "_seq")]
        from _seq.dataread import seq_split  # noqa: E402
        X, y = seq_split(df)                 # keeps patient_id+date for the windower
        bundle = _to_cpu(joblib.load(md / "model.joblib"))
        p = bundle.predict_proba(X)[:, 1]
    else:
        X, y = prep_split(df)
        bundle = _to_cpu(joblib.load(md / "model.joblib"))
        if d["addition"] == "0":
            sys.path.insert(0, str(EXP / "0"))
            from _model_architecture.stacked_2xgb_meta_lr.model import calibrated_proba  # noqa: E402
            p = calibrated_proba(bundle, X)
        else:  # Addition 1 TabPFN: sklearn predict_proba (loads on the GPU)
            p = bundle.predict_proba(X)[:, 1]

    np.savez(out, y=np.asarray(y, dtype=float), p=np.asarray(p, dtype=float), pid=pid)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
