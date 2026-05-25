"""Figure C2 - within-person discrimination (the load-bearing finding).

Per-patient hold-out AUROC for the headline discriminator (TabPFN v3,
full_features, chronological), pooled across the 5 expanding-window CV folds so
every patient with >=5 positive days is estimable (the late-enrolment hold-out is
too sparse for per-patient estimates). Each marker is one patient with its
Hanley-McNeil 95% CI; the reference lines are chance (0.5), the pooled AUROC, and
the precision-weighted within-person C-statistic (DerSimonian-Laird random
effects) with its CI band. The pooled AUROC sits far above a per-patient centre
near chance: pooled discrimination is mostly between-patient base-rate separation,
not within-person day-to-day ranking.

Usage: python fig_c2_within_person.py   (refits per CV fold; minutes, GPU for TabPFN)
"""
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

CV_WORKER = EXP / "5" / "_personal" / "_cv_oof_worker.py"
MIN_POS = 5


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
            print("  FAIL", leaf.name, (r.stderr.strip().splitlines() or ["?"])[-1])
            return None
        z = np.load(out)
        return z["y"].astype(float), z["p"].astype(float), z["pid"].astype(str)
    finally:
        out.unlink(missing_ok=True)


def _panel(ax, tgt, y, p, pid):
    scores = WP.per_patient_scores(y, p, pid, MIN_POS)
    est = scores[scores["estimable"]].sort_values("auroc").reset_index(drop=True)
    within = WP.within_person_cstatistic(scores)
    pooled = float(roc_auc_score(y, p))
    x = np.arange(len(est))
    err = np.array([1.96 * np.sqrt(WP._hanley_mcneil_var(
        r.auroc, int(r.n_pos), int(r.n - r.n_pos))) for r in est.itertuples()])
    ax.errorbar(x, est["auroc"], yerr=err, fmt="o", ms=3, lw=0.5,
                color=S.TARGET[tgt], alpha=0.6, ecolor="#cccccc", capsize=0)
    ax.axhspan(within["ci_low"], within["ci_high"], color="grey", alpha=0.18, zorder=0)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", label="chance (0.5)")
    ax.axhline(within["estimate"], color="#444444", lw=1.5,
               label=f"within-person C {within['estimate']:.2f}")
    ax.axhline(pooled, color=S.ARCH["TabPFN"], lw=1.6,
               label=f"pooled AUROC {pooled:.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"patient, sorted by AUROC (k={len(est)})")
    ax.set_ylabel("per-patient AUROC")
    ax.set_title(f"{tgt} - TabPFN, CV out-of-fold")
    ax.legend(loc="lower right")
    print(f"  {tgt:<9} pooled {pooled:.3f} | within {within['estimate']:.3f} "
          f"[{within['ci_low']:.3f}-{within['ci_high']:.3f}] | k={len(est)}")


def main():
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, tgt in zip(axes, ("headache", "migraine")):
        leaf = EXP / "1" / tgt / "full_features/tabpfn/version_3-default/70_15_15/chrono"
        r = _cv_predict(leaf)
        if r is not None:
            _panel(ax, tgt, *r)
    fig.suptitle("Per-patient discrimination vs pooled AUROC (history features, chrono)",
                 y=1.02)
    print("saved", S.save(fig, HERE / "figures" / "fig_c2_within_person"))


if __name__ == "__main__":
    main()
