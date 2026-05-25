"""Figure C1 - leakage decomposition.

Stratified-minus-chronological hold-out AUROC against the number of history
(lag/rolling) features in each feature set, holding the architecture fixed
(stacked_2xgb_meta_lr, NonHP) and averaging over the train/val/test ratios. The
optimism appears only for feature sets that carry history features (full, spano)
and is absent - slightly negative - for the no-history sets (no_rolling, park),
showing the stratified inflation is a feature-channel leak, not a model artefact.
This reproduces the table in results_findings.md Section 1 as a figure.

Usage: python fig_c1_leakage.py   (reads the latest experiment/comparison_*.csv)
"""
import csv
from pathlib import Path
from statistics import mean

import _figstyle as S
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP = REPO / "experiment"

# History (lag/rolling) feature count per set, verified in results_findings.md S1.
LAG = {"full_features": 17, "spano_features": 14,
       "no_rolling_features": 0, "park_features": 0}
LABEL = {"full_features": "full", "spano_features": "spano",
         "no_rolling_features": "no_rolling", "park_features": "park"}


def _auroc(s):
    try:
        return float(str(s).split("[")[0].strip())
    except (ValueError, AttributeError):
        return None


def _latest_csv():
    return max(EXP.glob("comparison_*.csv"), key=lambda p: p.stat().st_mtime)


def main():
    S.apply()
    csv_path = _latest_csv()
    agg = {}
    for x in csv.DictReader(open(csv_path)):
        if x["architecture"] != "stacked_2xgb_meta_lr":
            continue
        if x["hyperparameter"] == "HyperparameterTuned":   # NonHP cells only
            continue
        if x["splittype"] not in ("chrono", "stratified"):
            continue
        a = _auroc(x["holdout_AUROC"])
        if a is not None:
            agg.setdefault((x["target"], x["feature_set"], x["splittype"]), []).append(a)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    print(f"source: {csv_path.name}")
    for tgt in ("headache", "migraine"):
        pts = []
        for fset, nlag in LAG.items():
            ch = agg.get((tgt, fset, "chrono"))
            st = agg.get((tgt, fset, "stratified"))
            if not ch or not st:
                continue
            delta = mean(st) - mean(ch)
            pts.append((nlag, delta, LABEL[fset]))
            print(f"  {tgt:<9} {fset:<20} strat-chrono = {delta:+.3f}  (n_lag={nlag})")
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=S.TARGET[tgt], label=tgt, lw=1.5, ms=6)
        for x, y, lab in pts:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color=S.TARGET[tgt])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("history (lag / rolling) features in set")
    ax.set_ylabel("stratified - chronological AUROC")
    ax.set_title("Stratified-split optimism scales with history features")
    ax.legend(title="target")
    print("saved", S.save(fig, HERE / "figures" / "fig_c1_leakage"))


if __name__ == "__main__":
    main()
