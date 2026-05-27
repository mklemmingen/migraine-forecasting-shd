"""Figure F2 - Dolan-More performance profiles for AUROC, within-family scope.

One curve per architecture, per target, over the full_features cells. The curve
at factor tau is the fraction of cells where that architecture is within tau of
the best architecture in the cell [Dolan & More 2002, Math. Prog.]. Curves that
rise fastest in the top-left dominate; flat curves are unstable across cells.
Uses the same coverage-filtered architecture set as the critical-difference
diagram so the two views describe one matched design - that filter removes
non-HP and cross-family entries (add0_stacked_NonHP, add1_tabpfn_NonHP,
add4_window_mlp, pooled_lr) because they are missing on at least one cell,
which makes f2 (like f1) a WITHIN-FAMILY profile of HP variants vs TabPFN
versions rather than a true cross-architecture test.

Data source: the latest experiment/comparison_*.csv written by
run_aggregate_results.py.

Usage: python fig_f2_performance_profile.py
"""
# Paper caption (for LaTeX):
#   Dolan-More performance profile (Dolan & More 2002, Math. Prog.). For each
#   architecture, the curve at tolerance tau is the fraction of full_features cells
#   where its AUROC is within a factor tau of the best architecture in that cell.
#   Curves rising fastest toward 1.0 (top-left) dominate; flatter curves solve fewer
#   cells near-optimally. Restricted to full_features so every cell carries the same
#   competitors (matched design); architectures with incomplete cell coverage are
#   excluded (noted on the figure).
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
_sys.path.insert(0, str(_EXP / "_eval"))
from _parsing import latest_comparison_csv, entries_from_csv
from _benchmark_visuals import prepare_benchmark, render_performance_profile

HERE = _Path(__file__).resolve().parent


def main():
    csv_path = latest_comparison_csv(_EXP)
    if csv_path is None:
        raise SystemExit("no comparison_*.csv in experiment/ - run run_aggregate_results.py first")
    entries = entries_from_csv(csv_path)
    print(f"  source {csv_path.name} ({len(entries)} leaves)")
    for tgt in ("headache", "migraine"):
        prep = prepare_benchmark(entries, "AUROC", tgt,
                                 restrict_feature_sets={"full_features"})
        if prep is None:
            print(f"  skip {tgt}: fewer than two matched-coverage architectures")
            continue
        out = HERE / "figures" / f"fig_f2_performance_profile_{tgt}"
        render_performance_profile(
            prep["matrix_kept"], prep["text_kept"], out,
            title=f"Performance profile (Dolan-More) - {tgt} AUROC",
            higher_is_better=prep["higher_is_better"],
            arch_tuples=prep["arch_kept"], dropped_archs=prep["dropped_text"],
        )
        print("saved", out)


if __name__ == "__main__":
    main()
