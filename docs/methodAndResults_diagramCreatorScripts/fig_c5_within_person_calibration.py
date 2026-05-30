"""Figure C5 - within-person calibration at the headline cells.

Adds the within-person calibration leg to the within-person discrimination
reporting in body §3.6. The within-person C-statistic at the headline cells
clusters at 0.53-0.57 (near chance) per §3.6; this figure asks the calibration
analogue: when a model ranks a given patient's days into within-patient
quartiles by predicted probability, do the highest-quartile days actually carry
the patient's observed positives?

For each estimable patient (per the Addition-5 ``MIN_POS = 5`` floor that
underlies the within-person C - patients need at least 5 positive days for a
per-patient AUROC to be reportable, so we use the same floor for the
calibration view), each day is assigned to one of four quartiles by its
predicted probability under the leaf's model. The mean predicted probability
and the mean observed positive rate per quartile are then averaged across
patients (patient-weighted pooling) to give four (predicted, observed) points
per architecture per target. A patient-bootstrap 95% CI is drawn on the
observed-rate point estimate of each quartile.

Reads as a sibling of fig_c4: fig_c4 carries pooled-level moderate calibration
(loess across all patient-days at once); fig_c5 carries within-person
calibration (within-patient quartiles, then pooled). The two together extend
body §3.4 (slope + CITL weak calibration at the pooled level) into the
moderate-calibration level at both pooling layers.

Usage: python fig_c5_within_person_calibration.py
"""
# §11 compliance (figure_design_requirements.md, reviewer-derived 2026-05-28):
#   §11.1 CIs on headline metric:        patient-bootstrap CI on per-quartile observed
#   §11.3 self-contained caption:        cohort name + n + CI method in suptitle
#   §11.6 CC BY 4.0 footer:              auto-applied by S.save()
#   §11.7 EPV-5.5 annotation:            S.epv_annotation() on migraine panel
#   §11.10 no "substantial"/"large":     verified
#   §11.11 self-check:                   this block
#   §6 shared legend:                    fig.legend frameless below panels
import importlib.util as _ilu
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path[0:0] = [str(EXP), str(EXP / "5" / "_personal")]
import _style as S  # noqa: E402
from within_person import MIN_POS, per_patient_scores  # noqa: E402

# Use the CV-OOF worker, NOT the hold-out predict worker - body §3.6 cites
# OOF-CV predictions ("Under out-of-fold cross-validation on the
# non-hyperparameter-tuned 70/30 chronological leaves...") because the
# hold-out test partition on 70/30 chrono is too late-enrolment-limited to
# give 5 positives per patient for most patients on the migraine target;
# the 5-fold expanding-window OOF predictions cover the full date range and
# produce the 57 / 19 estimable patient counts named in §3.6.
WORKER = EXP / "5" / "_personal" / "_cv_oof_worker.py"

N_QUARTILES = 4
N_BOOT = 500


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _predict(leaf: Path):
    addition = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"c5_{uuid.uuid4().hex}.npz"
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
        return (
            z["y"].astype(int),
            z["p"].astype(float),
            z["pid"].astype(str),
        )
    finally:
        out.unlink(missing_ok=True)


def _per_patient_quartile_means(y: np.ndarray, p: np.ndarray, pid: np.ndarray) -> pd.DataFrame:
    """Per-patient per-quartile mean(predicted) and mean(observed).

    Estimable patients are filtered to the same ``MIN_POS = 5`` floor that the
    within-person C-statistic uses, so the figure's denominator matches the
    19 / 57 estimable counts already reported in body §3.6. Patients whose
    per-patient predictions cannot be cut into 4 distinct quantile bins (too
    many ties or fewer than 4 unique predicted values) are dropped from the
    quartile aggregation only.
    """
    scores = per_patient_scores(y, p, pid, MIN_POS)
    estimable_pids = scores.loc[scores["estimable"], "patient"].astype(str).tolist()

    rows = []
    for ppid in estimable_pids:
        mask = pid == ppid
        py, pp = y[mask], p[mask]
        try:
            q = pd.qcut(pp, N_QUARTILES, labels=False, duplicates="drop")
        except ValueError:
            continue
        q = np.asarray(q)
        if len(set(q[~pd.isna(q)])) < N_QUARTILES:
            continue
        for k in range(N_QUARTILES):
            qm = q == k
            if qm.sum() == 0:
                continue
            rows.append({
                "patient": ppid,
                "quartile": k,
                "mean_p": float(pp[qm].mean()),
                "mean_y": float(py[qm].mean()),
            })
    return pd.DataFrame(rows)


def _pool_with_patient_bootstrap_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Patient-weighted pooled per-quartile means with patient-bootstrap 95% CI.

    Patient-bootstrap (resample whole patients with replacement) rather than
    row-bootstrap because the unit being aggregated is the patient - the same
    discipline body §2.8 acknowledges is the appropriate resampling unit for
    within-person quantities."""
    pooled = (df.groupby("quartile")[["mean_p", "mean_y"]]
              .mean()
              .rename(columns={"mean_p": "pred", "mean_y": "obs"})
              .reset_index())

    patients = df["patient"].unique()
    rng = np.random.default_rng(42)
    boot_obs = np.empty((N_BOOT, N_QUARTILES))
    boot_pred = np.empty((N_BOOT, N_QUARTILES))
    for b in range(N_BOOT):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        boot_df = pd.concat([df[df["patient"] == p] for p in sampled])
        means = boot_df.groupby("quartile")[["mean_p", "mean_y"]].mean()
        for k in range(N_QUARTILES):
            boot_pred[b, k] = means.loc[k, "mean_p"] if k in means.index else np.nan
            boot_obs[b, k] = means.loc[k, "mean_y"] if k in means.index else np.nan

    pooled["obs_ci_low"] = np.nanpercentile(boot_obs, 2.5, axis=0)
    pooled["obs_ci_high"] = np.nanpercentile(boot_obs, 97.5, axis=0)
    pooled["pred_ci_low"] = np.nanpercentile(boot_pred, 2.5, axis=0)
    pooled["pred_ci_high"] = np.nanpercentile(boot_pred, 97.5, axis=0)
    pooled["n_patients"] = len(patients)
    return pooled


def _resolve_headlines() -> dict[str, list[tuple[str, Path]]]:
    """Same composite-tracked leaf resolution as fig_c3, fig_c4 and the T2-5
    sensitivity runner. Add-4 window-MLP stays pinned per fig_c3."""
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


def main() -> None:
    S.apply()
    leaves = _resolve_headlines()
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.2))
    for ax, ltr in zip(axes.ravel(), "ab"):
        S.panel_label(ax, ltr)
    mark = {"XGBoost stack": "o", "TabPFN": "s", "window-MLP": "^"}

    for ax, tgt in zip(axes, ("headache", "migraine")):
        hi_lim = 0.0
        n_pat_by_arch: list[tuple[str, int]] = []
        for label, leaf in leaves[tgt]:
            r = _predict(leaf)
            if r is None:
                continue
            y, p, pid = r
            per_patient = _per_patient_quartile_means(y, p, pid)
            if per_patient.empty:
                print(f"  {tgt:<9} {label:<13} 0 estimable patients - skip")
                continue
            pooled = _pool_with_patient_bootstrap_ci(per_patient)
            col = S.arch_color(label)
            mk = mark.get(label, "o")
            ax.errorbar(pooled["pred"], pooled["obs"],
                        yerr=[pooled["obs"] - pooled["obs_ci_low"],
                              pooled["obs_ci_high"] - pooled["obs"]],
                        color=col, lw=1.5, marker=mk, markersize=6,
                        capsize=3, elinewidth=0.8, label=label)
            hi_lim = max(hi_lim, float(pooled["pred"].max()), float(pooled["obs_ci_high"].max()))
            n_pat = int(pooled["n_patients"].iloc[0])
            n_pat_by_arch.append((label, n_pat))
            for _, row in pooled.iterrows():
                print(f"  {tgt:<9} {label:<13} Q{int(row['quartile'])+1}  "
                      f"pred {row['pred']:.3f}  obs {row['obs']:.3f} "
                      f"[{row['obs_ci_low']:.3f}, {row['obs_ci_high']:.3f}]")

        ax.plot([0, 1], [0, 1], color=S.REF_COLOR, lw=S.REF_LW, ls="--", label="perfect")
        lim = min(1.0, hi_lim * 1.15 + 0.02)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("pooled mean predicted (per-patient quartile)")
        ax.set_ylabel("pooled mean observed (per-patient quartile)")
        ax.set_title(tgt)
        S.epv_annotation(ax, tgt, cell="full_features", loc="upper left")
        if n_pat_by_arch:
            uniq = sorted(set(n for _, n in n_pat_by_arch))
            note = (f"n_patients (estimable, ≥{MIN_POS} pos): "
                    + " / ".join(str(n) for n in uniq))
            ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7, color=S.GREY)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Within-person calibration - Park 2016 SHD, n=62\n"
                 f"per-patient quartiles by predicted probability, patient-bootstrap 95% CI on observed",
                 y=1.04, fontsize=10)
    print("saved", S.save(fig, HERE / "figures" / "fig_c5_within_person_calibration"))


if __name__ == "__main__":
    main()
