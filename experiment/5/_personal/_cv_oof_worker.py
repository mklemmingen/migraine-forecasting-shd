"""CV out-of-fold prediction worker for the Addition 5 within-person evaluation.

Refits a leaf's architecture across the 5 expanding-window TimeSeriesSplit folds
(diary_cv5_timeseries.parquet) and pools the held-out-fold predictions, so every
patient receives out-of-sample predictions across the full date range - far more
per-patient positives than the late-enrolment-limited hold-out, which is what
makes the within-person C-statistic estimable for the migraine target.

Like the hold-out worker, it runs in a fresh interpreter per leaf so the
conflicting ``_model_architecture`` packages and devices never collide. The
driver runs Addition 1 (TabPFN) with the GPU visible and Additions 0/4 with it
hidden.

Usage: python _cv_oof_worker.py <leaf_model_dir> <out_npz>
"""
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]
REPO = EXP.parent
RATIOS = ("70_15_15", "70_30", "80_20")
SPLITS = ("chrono", "stratified", "patient")
N_SPLITS = 5


def _dims(model_dir: Path) -> dict:
    parts = model_dir.relative_to(EXP).parts
    return {"addition": parts[0], "target": parts[1], "feature_set": parts[2]}


def main(model_dir: str, out: str) -> None:
    import pandas as pd
    md = Path(model_dir).resolve()
    d = _dims(md)
    sys.path.insert(0, str(EXP))
    from _dataRead.read import load_raw, prep_split, chronological_subsplit, TARGET_COL  # noqa: E402
    from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
    from _dataRead.filter_to_park_features import select_park_features  # noqa: E402
    loaders = {"full_features": None,
               "no_rolling_features": select_non_rolling_features,
               "park_features": select_park_features}

    add = d["addition"]
    if add == "0":
        sys.path.insert(0, str(EXP / "0"))
        from _model_architecture.stacked_2xgb_meta_lr.model import build_model, calibrated_proba  # noqa: E402

        def fit_predict(train_fold, val_fold):
            tr, ca = chronological_subsplit(train_fold)
            xtr, ytr = prep_split(tr)
            xca, yca = prep_split(ca)
            xv, _ = prep_split(val_fold)
            bundle = build_model(xtr, ytr, xca, yca)
            return calibrated_proba(bundle, xv)
    elif add == "1":
        sys.path.insert(0, str(EXP / "1"))
        parts = md.relative_to(EXP).parts
        version = parts[4] if len(parts) > 4 else ""
        if version == "version_3-default":
            from _model_architecture.tabpfn_v3.model import build_tabpfn_v3 as _builder  # noqa: E402
        elif version == "version_3-binary":
            from _model_architecture.tabpfn_v3_binary.model import build_tabpfn_v3_binary as _builder  # noqa: E402
        elif version == "version_2-6":
            from _model_architecture.tabpfn.model import build_tabpfn as _builder  # noqa: E402
        elif version == "version_2-5-finetuned":
            from _model_architecture.finetunedtabpfn_v2_5.model import build_finetunedtabpfn as _builder  # noqa: E402
        elif version == "version_2-5-auto":
            from _model_architecture.autotabpfn_v2_5.model import build_autotabpfn as _builder  # noqa: E402
        elif version == "version_2-5-real":
            from _model_architecture.realtabpfn.model import build_realtabpfn as _builder  # noqa: E402
        else:
            raise SystemExit(f"unknown tabpfn version {version!r} for leaf {md}")

        def fit_predict(train_fold, val_fold):
            xtr, ytr = prep_split(train_fold)
            xv, _ = prep_split(val_fold)
            return _builder(xtr, ytr).predict_proba(xv)[:, 1]
    else:  # sequence (Addition 4)
        sys.path[0:0] = [str(EXP / "4"), str(EXP / "4" / "_seq")]
        from _model_architecture.window_mlp.model import build_window_mlp  # noqa: E402
        from _seq.dataread import seq_split  # noqa: E402

        def fit_predict(train_fold, val_fold):
            xtr, ytr = seq_split(train_fold)
            xv, _ = seq_split(val_fold)
            return build_window_mlp(xtr, ytr).predict_proba(xv)[:, 1]

    cv = load_raw(str(REPO / "data" / "processed" / d["target"] / "diary_cv5_timeseries.parquet"),
                  loader=loaders.get(d["feature_set"]))
    ys, ps, pids = [], [], []
    for fold in range(1, N_SPLITS + 1):
        train_fold = cv[cv["cv_fold"] < fold].copy()
        val_fold = cv[cv["cv_fold"] == fold].copy()
        if train_fold.empty or val_fold.empty:
            continue
        ps.append(np.asarray(fit_predict(train_fold, val_fold), dtype=float))
        ys.append(val_fold[TARGET_COL].to_numpy(dtype=float))
        pids.append(val_fold["patient_id"].to_numpy().astype(str))

    np.savez(out, y=np.concatenate(ys), p=np.concatenate(ps), pid=np.concatenate(pids))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
