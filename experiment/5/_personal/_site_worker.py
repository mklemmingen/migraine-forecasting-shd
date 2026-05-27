"""Leave-one-site-out refit/predict worker for the Route A external validation.

Mirrors _cv_oof_worker.py, but the split is geographic rather than temporal: the
model is refit on the union of the OTHER recruitment site's days and applied to
the held-out site, whose patients are therefore entirely unseen in training. A
chronological cal sub-split of the TRAINING site supplies the operating threshold
(the held-out site is never used to tune it), matching the rest of the benchmark.

Runs in a fresh interpreter per leaf so the conflicting ``_model_architecture``
packages and the TabPFN GPU/CPU split never collide; the driver gives Addition 1
the GPU and hides it from Additions 0/4.

Usage: python _site_worker.py <leaf_model_dir> <held_out_site> <out_npz>
"""
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]
REPO = EXP.parent
HERE = Path(__file__).resolve().parent


def _dims(model_dir: Path) -> dict:
    parts = model_dir.relative_to(EXP).parts
    return {"addition": parts[0], "target": parts[1], "feature_set": parts[2]}


def main(model_dir: str, held_out: str, out: str) -> None:
    md = Path(model_dir).resolve()
    d = _dims(md)
    sys.path[0:0] = [str(HERE), str(EXP)]   # HERE first so _site resolves before stdlib
    from _dataRead.read import load_raw, prep_split, chronological_subsplit, TARGET_COL  # noqa: E402
    from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
    from _dataRead.filter_to_park_features import select_park_features  # noqa: E402
    import _site as SITE  # noqa: E402
    loaders = {"full_features": None,
               "no_rolling_features": select_non_rolling_features,
               "park_features": select_park_features}

    add = d["addition"]
    if add == "0":
        sys.path.insert(0, str(EXP / "0"))
        from _model_architecture.stacked_2xgb_meta_lr.model import build_model, calibrated_proba  # noqa: E402

        def fit_predict(tr, ca, te):
            xtr, ytr = prep_split(tr); xca, yca = prep_split(ca); xte, _ = prep_split(te)
            b = build_model(xtr, ytr, xca, yca)
            return calibrated_proba(b, xca), calibrated_proba(b, xte)
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

        def fit_predict(tr, ca, te):
            xtr, ytr = prep_split(tr); xca, _ = prep_split(ca); xte, _ = prep_split(te)
            m = _builder(xtr, ytr)
            return m.predict_proba(xca)[:, 1], m.predict_proba(xte)[:, 1]
    else:  # sequence (Addition 4)
        sys.path[0:0] = [str(EXP / "4"), str(EXP / "4" / "_seq")]
        from _model_architecture.window_mlp.model import build_window_mlp  # noqa: E402
        from _seq.dataread import seq_split  # noqa: E402

        def fit_predict(tr, ca, te):
            xtr, ytr = seq_split(tr); xca, _ = seq_split(ca); xte, _ = seq_split(te)
            m = build_window_mlp(xtr, ytr)
            return m.predict_proba(xca)[:, 1], m.predict_proba(xte)[:, 1]

    df = load_raw(str(REPO / "data" / "processed" / d["target"] / "diary_cv5_timeseries.parquet"),
                  loader=loaders.get(d["feature_set"]))
    df = SITE.attach_site(df).dropna(subset=["site"])
    # 'site' is the split key only; drop it before any feature matrix is built so
    # the string column never reaches XGBoost / the window-MLP (prep_split and
    # seq_split drop the id/target columns but not this one).
    train = df[df["site"] != held_out].drop(columns=["site"]).copy()
    test = df[df["site"] == held_out].drop(columns=["site"]).copy()
    tr, ca = chronological_subsplit(train)
    p_val, p_test = fit_predict(tr, ca, test)
    np.savez(out,
             y_val=ca[TARGET_COL].to_numpy(dtype=float), p_val=np.asarray(p_val, dtype=float),
             y_test=test[TARGET_COL].to_numpy(dtype=float), p_test=np.asarray(p_test, dtype=float),
             pid_test=test["patient_id"].to_numpy().astype(str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
