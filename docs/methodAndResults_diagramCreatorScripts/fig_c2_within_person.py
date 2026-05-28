"""Figure C2 - within-person discrimination (the load-bearing finding).

Per-patient hold-out AUROC for the TabPFN leaf in each target's headline cell
(full_features, chronological), pooled across the 5 expanding-window CV folds so
every patient with >=5 positive days is estimable. Each marker is one patient
with its Hanley-McNeil 95% CI; reference lines are chance (0.5), the pooled
AUROC with its bootstrap 95% CI band, and the precision-weighted within-person
C-statistic (DerSimonian-Laird random effects) with its CI band. The pooled
AUROC sits far above a per-patient centre near chance: pooled discrimination
is mostly between-patient base-rate separation, not within-person day-to-day
ranking.

Usage: python fig_c2_within_person.py   (refits per CV fold; minutes, GPU for TabPFN)
"""
# §11 compliance (figure_design_requirements.md, reviewer-derived 2026-05-28):
#   §11.1 CIs on headline metric:        Hanley-McNeil per-patient whiskers; bootstrap
#                                        CI band on pooled AUROC ref line; DL CI band on
#                                        within-person C-stat ref line
#   §11.2 estimability denominators:     k=len(est) annotated in xlabel + on panel
#   §11.3 self-contained caption:        S.caption_block() composed for suptitle
#   §11.6 CC BY 4.0 footer:              S.cc_by_footer() invoked
#   §11.7 EPV-5.5 annotation:            S.epv_annotation() on migraine panel
#   §11.10 no "substantial"/"large":     verified in captions
#   §11.11 self-check:                   this block
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "experiment"))
import _style as S
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"
sys.path.insert(0, str(EXP / "5" / "_personal"))
import within_person as WP  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# Reuse the figdata loader the g/h scripts use so leaf-path resolution stays
# in lockstep with the composite_sorted selection in experiment/2/select.py.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_exp2_figures", EXP / "2" / "_figures.py")
_F = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_F)

CV_WORKER = EXP / "5" / "_personal" / "_cv_oof_worker.py"
MIN_POS = 5


def _resolve_tabpfn_leaf(headlines, target):
    """Return the figdata-tracked TabPFN leaf for the headline cell
    (full_features/chrono) of ``target``: prefer the headline if family=tabpfn,
    otherwise the runner-up. Returns None when neither row is tabpfn (e.g. if
    figdata pre-dates the leaf_dir field)."""
    for role in ("headline", "runner_up"):
        for e in headlines:
            if (e.get("target") == target and e.get("feature_set") == "full_features"
                    and e.get("splittype") == "chrono" and e.get("role") == role
                    and e.get("family") == "tabpfn" and e.get("leaf_dir")):
                return Path(e["leaf_dir"])
    return None


def _env(addition: str) -> dict:
    e = os.environ.copy()
    if addition != "1":
        e["CUDA_VISIBLE_DEVICES"] = ""
        e["HIP_VISIBLE_DEVICES"] = ""
    return e


def _cv_predict(leaf: Path):
    add = leaf.relative_to(EXP).parts[0]
    out = Path(tempfile.gettempdir()) / f"c2_{uuid.uuid4().hex}.npz"
    try:
        r = subprocess.run([sys.executable, str(CV_WORKER), str(leaf), str(out)],
                           capture_output=True, text=True, timeout=3600, env=_env(add))
        if r.returncode != 0 or not out.exists():
            print(f"  FAIL {leaf.name} (returncode={r.returncode}, out_exists={out.exists()})")
            if r.stderr:
                print("  --- worker stderr ---")
                for line in r.stderr.strip().splitlines():
                    print(f"    {line}")
                print("  --- end stderr ---")
            return None
        z = np.load(out)
        return z["y"].astype(float), z["p"].astype(float), z["pid"].astype(str)
    finally:
        out.unlink(missing_ok=True)


def _bootstrap_pooled_auroc_ci(y, p, n_boot=1000, seed=42):
    """Patient-day bootstrap CI on pooled AUROC. Matches body §2.8
    methods (patient-day resampling; the under-coverage caveat against
    patient-cluster bootstrap is disclosed in the body Methods)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if 0 < ys.sum() < n:
            aucs.append(roc_auc_score(ys, p[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _panel(ax, tgt, y, p, pid):
    scores = WP.per_patient_scores(y, p, pid, MIN_POS)
    est = scores[scores["estimable"]].sort_values("auroc").reset_index(drop=True)
    within = WP.within_person_cstatistic(scores)
    pooled = float(roc_auc_score(y, p))
    pooled_lo, pooled_hi = _bootstrap_pooled_auroc_ci(y, p)
    x = np.arange(len(est))
    err = np.array([1.96 * np.sqrt(WP._hanley_mcneil_var(
        r.auroc, int(r.n_pos), int(r.n - r.n_pos))) for r in est.itertuples()])
    ax.errorbar(x, est["auroc"], yerr=err, fmt="o", ms=3, lw=0.5,
                color=S.TARGET[tgt], alpha=0.6, ecolor=S.FAINT, capsize=0)
    ax.axhspan(within["ci_low"], within["ci_high"], color=S.GREY,
               alpha=S.CI_ALPHA, zorder=0)
    ax.axhspan(pooled_lo, pooled_hi, color=S.ARCH["TabPFN"],
               alpha=0.12, zorder=0)
    S.refline(ax, y=0.5, label="chance (0.5)")
    ax.axhline(within["estimate"], color=S.SOFT, lw=1.5,
               label=f"within-person C {within['estimate']:.2f} "
                     f"[{within['ci_low']:.2f}-{within['ci_high']:.2f}]")
    ax.axhline(pooled, color=S.ARCH["TabPFN"], lw=1.6,
               label=f"pooled AUROC {pooled:.2f} "
                     f"[{pooled_lo:.2f}-{pooled_hi:.2f}]")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"patient, sorted by AUROC ({len(est)} of 63 estimable)")
    ax.set_ylabel("per-patient AUROC")
    ax.set_title(f"{tgt} - TabPFN, CV out-of-fold")
    ax.legend(loc="upper left")
    S.epv_annotation(ax, tgt, cell="full_features", loc="lower right")
    print(f"  {tgt:<9} pooled {pooled:.3f} [{pooled_lo:.3f}-{pooled_hi:.3f}] "
          f"| within {within['estimate']:.3f} "
          f"[{within['ci_low']:.3f}-{within['ci_high']:.3f}] | k={len(est)}")


def main():
    S.apply()
    figdata_path = _F.latest_figdata(EXP / "2")
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    headlines = _F.load_figdata(figdata_path).get("headlines", [])
    print(f"  source {figdata_path.name} ({len(headlines)} headline rows)")
    fig, axes = plt.subplots(1, 2, figsize=S.figsize("double", 4.6))
    for _ax, _lt in zip(axes.ravel(), "abcdefgh"):
        S.panel_label(_ax, _lt)
    for ax, tgt in zip(axes, ("headache", "migraine")):
        leaf = _resolve_tabpfn_leaf(headlines, tgt)
        if leaf is None:
            print(f"  no tabpfn entry for {tgt} in figdata; skipping panel "
                  "(re-run experiment/2/compare.py to refresh leaf_dir)")
            continue
        print(f"  {tgt} TabPFN leaf: {leaf.relative_to(EXP)}")
        r = _cv_predict(leaf)
        if r is not None:
            _panel(ax, tgt, *r)
            # Two-line title so the bold panel-label (a / b) at the top-left
            # does not collide with the long 4-token slug stem.
            ax.set_title(f"{tgt}\n{S.leaf_slug(leaf)}", fontsize=10)
    fig.suptitle("Per-patient discrimination vs pooled AUROC\n"
                 "Park 2016 SHD cohort, n=62; CV out-of-fold; "
                 "Hanley-McNeil 95% CI per patient; bootstrap CI on pooled",
                 y=1.04, fontsize=10)
    S.cc_by_footer(fig)
    print("saved", S.save(fig, HERE / "figures" / "fig_c2_within_person"))


if __name__ == "__main__":
    main()
