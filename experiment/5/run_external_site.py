"""Two-fold leave-one-site-out external validation driver (Route A).

Geographic internal-external validation across the two SHD recruitment sites
[steyerberg2016validation, p. 245]: for each held-out site, every model is refit
on the other site and applied to the held-out site's unseen patients. Three
transport questions per fold (docs/external_validation_site.md Section 3):

  1. discrimination transport - pooled AUROC/AUPRC, in the standard bootstrap
     contract folded into comparison_*.html (datasplit=external,
     splittype=site_<held>);
  2. within-person transport - the precision-weighted within-person C-statistic
     on the held-out site (reusing _personal/within_person.py), testing whether
     the near-chance per-patient result replicates off-site;
  3. calibration transport - calibration-in-the-large and slope on the held-out
     site [huang2020calibration, p. 621]. Because the sites differ in base rate, a
     model trained on one must mis-set its mean predicted risk on the other, and
     that intercept drift is the distinctive transportability result.

The LR regimes run in-process; the architectures (Additions 0/1/4) refit through
_personal/_site_worker.py in a per-addition subprocess (the GPU is shown only to
Addition 1). Imbalance handling is unchanged (no reweighting; external threshold).
"""
import datetime as _dt
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent      # experiment/5/
EXP = HERE.parent
REPO = EXP.parent
sys.path[0:0] = [str(EXP), str(HERE / "_personal")]
import within_person as WP  # noqa: E402
import regimes as RG  # noqa: E402
import _site as SITE  # noqa: E402
from _dataRead.read import (  # noqa: E402
    load_raw, chronological_subsplit, NON_FEATURE_COLS, TARGET_COL)
from _dataRead.filter_to_no_rolling_features import select_non_rolling_features  # noqa: E402
from _dataRead.filter_to_park_features import select_park_features  # noqa: E402
from _eval.metrics_lib import calibration_slope  # noqa: E402

SITE_WORKER = HERE / "_personal" / "_site_worker.py"
LOADERS = {"full_features": None,
           "no_rolling_features": select_non_rolling_features,
           "park_features": select_park_features}
ARCH_LABEL = {"0": "add0_stacked", "1": "add1_tabpfn", "4": "add4_window_mlp"}


def _env(addition: str) -> dict:
    env = os.environ.copy()
    if addition != "1":  # GPU only for TabPFN; CPU for XGBoost/sequence
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
    return env


def calibration_transport(y, p) -> dict:
    """Calibration-in-the-large (mean observed vs mean predicted) and slope on a
    held-out site. citl<0 and oe_ratio<1 both denote over-prediction - the
    expected sign when a higher-base-rate site's model is applied to a lower one."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    obs, pred = float(y.mean()), float(p.mean())
    return {"observed_rate": round(obs, 4), "mean_predicted": round(pred, 4),
            "oe_ratio": round(obs / pred, 3) if pred > 0 else float("nan"),
            "citl": round(obs - pred, 4), "cal_slope": round(calibration_slope(y, p), 3)}


def _arch_predict(leaf: Path, held: str, addition: str):
    out = Path(tempfile.gettempdir()) / f"siteA_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(SITE_WORKER), str(leaf), held, str(out)],
                           capture_output=True, text=True, timeout=2400, env=_env(addition))
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr.strip().splitlines() or ["worker failed"])[-1]
            print(f"    SKIP {ARCH_LABEL[addition]} site_{held}: {tail}")
            return None
        z = np.load(out, allow_pickle=True)
        return {k: z[k] for k in z.files}
    finally:
        out.unlink(missing_ok=True)


def _record(rows, model, target, feature_set, held, train_rate,
            y_val, p_val, y_test, p_test, pid):
    """Emit the standard contract and append the transport-metrics row."""
    out_dir = HERE / target / feature_set / model / "external" / f"site_{held}"
    RG.emit_holdout_results(str(out_dir), f"ROUTE A / {feature_set} / {model} / site_{held}",
                            pd.Series(np.asarray(y_val, dtype=float)), np.asarray(p_val, dtype=float),
                            pd.Series(np.asarray(y_test, dtype=float)), np.asarray(p_test, dtype=float))
    y = np.asarray(y_test, dtype=float); p = np.asarray(p_test, dtype=float)
    pid = np.asarray(pid).astype(str)
    within = WP.within_person_cstatistic(WP.per_patient_scores(y, p, pid, WP.MIN_POS))
    cal = calibration_transport(y, p)
    row = {"target": target, "feature_set": feature_set, "model": model, "held_out": held,
           "n_test": int(len(y)), "train_rate": round(train_rate, 4),
           "auroc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else float("nan"),
           "auprc": round(float(average_precision_score(y, p)), 4),
           "within_cstat": round(float(within["estimate"]), 3) if within["estimate"] == within["estimate"] else float("nan"),
           "within_k": int(within["k_estimable"]), **cal}
    rows.append(row)
    print(f"  {model:<16} site_{held:<10} AUROC {row['auroc']:.3f} | within {row['within_cstat']} "
          f"(k={row['within_k']}) | O:E {cal['oe_ratio']} slope {cal['cal_slope']} "
          f"(pred {cal['mean_predicted']:.3f} vs obs {cal['observed_rate']:.3f})")
    return row


def default_leaves() -> dict:
    """{target -> [Add 0/1/4 leaf model dirs]} for the full_features chrono cells."""
    out = {}
    for tgt in ("headache", "migraine"):
        ls = []
        for m in (EXP / "0" / tgt / "full_features").rglob("NonHP/model.joblib"):
            if "stacked_2xgb" in str(m) and "/70_15_15/chrono/" in str(m):
                ls.append(m.parent); break
        for sub in ("1/{t}/full_features/tabpfn/version_3-default/70_15_15/chrono",
                    "4/{t}/full_features/sequence/version_window-mlp/70_15_15/chrono"):
            dd = EXP / sub.format(t=tgt)
            if (dd / "model.joblib").exists():
                ls.append(dd)
        out[tgt] = ls
    return out


def run_cell(target: str, feature_set: str, leaves: list) -> list:
    df = SITE.attach_site(load_raw(
        str(REPO / "data" / "processed" / target / "diary_cv5_timeseries.parquet"),
        loader=LOADERS.get(feature_set))).dropna(subset=["site"])
    fc = [c for c in df.columns if c not in NON_FEATURE_COLS and c != "site"]
    rows = []
    print(f"\n=== {target} / {feature_set} (sites: "
          + ", ".join(f"{s}={int((df.site == s).sum())}r/"
                      f"{df.loc[df.site == s, TARGET_COL].mean():.1%}" for s in SITE.SITES) + ") ===")
    for held in SITE.SITES:
        train = df[df["site"] != held]; test = df[df["site"] == held]
        if train.empty or test.empty:
            continue
        tr, ca = chronological_subsplit(train)
        train_rate = float(train[TARGET_COL].mean())
        # Under patient-disjoint site validation the held-out site's patients are
        # entirely unseen, so per_patient and partial_pool have nothing learned
        # for them and collapse exactly to the pooled global model; only pooled is
        # run (its row stands in for all three, and that collapse is itself the
        # cold-start result - personalisation cannot transfer to a new patient).
        p_val, p_test = RG.pooled(tr, ca, test, fc)
        _record(rows, "pooled_lr", target, feature_set, held, train_rate,
                ca[TARGET_COL], p_val, test[TARGET_COL], p_test, test["patient_id"])
        for leaf in leaves:
            add = leaf.relative_to(EXP).parts[0]
            z = _arch_predict(leaf, held, add)
            if z is None:
                continue
            _record(rows, ARCH_LABEL[add], target, feature_set, held, train_rate,
                    z["y_val"], z["p_val"], z["y_test"], z["p_test"], z["pid_test"])
    return rows


def main(feature_set: str = "full_features"):
    leaves = default_leaves()
    rows = []
    for target in ("headache", "migraine"):
        rows += run_cell(target, feature_set, leaves.get(target, []))
    if rows:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = HERE / f"external_site_summary_{ts}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved summary: {out}")
        print("Site-holdout cells emit the standard contract -> comparison_*.html "
              "(datasplit=external, splittype=site_<held>)")


if __name__ == "__main__":
    main()
