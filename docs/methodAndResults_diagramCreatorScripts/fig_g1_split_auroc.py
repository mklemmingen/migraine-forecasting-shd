"""Figure G1 - discrimination by split type, per cell (Addition 2).

Best hold-out AUROC of the COMPOSITE-SELECTED headline model per (target,
feature-set) cell, one bar per split strategy with 95% CI whiskers; bars
whose CI reaches chance are hatched. The chronological bar is the
deployable forecast, stratified is the leakage contrast, patient is
generalisation to unseen patients.

Caveat for the results_findings.md §2 asymmetric-hierarchy claim
(headache stratified > patient > chrono; migraine stratified > chrono
> patient): §2 fixes the architecture (stacked_2xgb NonHP) and averages
across ratios, while this figure plots the composite-selected headline
per cell, which mixes architectures (XGB on some cells, TabPFN on
others). The per-cell orderings in this figure therefore need NOT match
the §2 hierarchy - a reader cross-referencing the two should know that
§2 is a within-architecture cohort summary and g1 is a cross-architecture
headline summary.

Data source: the latest experiment/2/figdata_*.json frozen by
experiment/2/compare.py, so the figure pins to one selection run.

Usage: python fig_g1_split_auroc.py
"""
# §11 compliance: AUROC bars per cell with 95% CI whiskers (already present).
#   §11.1 PASS (CI whiskers); §11.3 caption: cohort+n in rendered title
#   §11.10 no banned adjectives; §11.11 self-check this block
# Paper caption (for LaTeX; the reference-line and hatch keys were moved off the
# axis label, which duplicated the legend):
#   Best hold-out AUROC of the headline model per (target, feature-set) cell, one bar
#   per split strategy with 95% CIs. The dotted line is chance (0.5); a hatched bar's
#   95% CI reaches chance (not significantly forecastable). Chronological is the
#   deployable forecast; stratified is leakage-inflated (a shuffled split places
#   adjacent days, which share history/rolling feature values, on both sides of the
#   train/test boundary); patient is generalisation to unseen patients.
import sys as _sys
import importlib.util as _ilu
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
# experiment/ on the path for _style; load _figures by file path rather than
# adding experiment/2 to sys.path, where its select.py shadows the stdlib select.
_sys.path.insert(0, str(_EXP))
_spec = _ilu.spec_from_file_location("_exp2_figures", _EXP / "2" / "_figures.py")
F = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(F)

HERE = _Path(__file__).resolve().parent
ADDITION2 = _EXP / "2"


def main():
    figdata_path = F.latest_figdata(ADDITION2)
    if figdata_path is None:
        raise SystemExit("no figdata_*.json in experiment/2/ - run experiment/2/compare.py first")
    data = F.load_figdata(figdata_path)
    print(f"  source {figdata_path.name}")
    out = HERE / "figures" / "fig_g1_split_auroc"
    if F.split_auroc_figure(data["headlines"], out) is None:
        print("  skip: no headline cells in the figure data")
    else:
        print("saved", out)


if __name__ == "__main__":
    main()
