"""Figure F1 - critical-difference diagrams (Friedman + Nemenyi) for AUROC.

One diagram per target over the full_features cells - the matched design the
Friedman test requires (every architecture evaluated on every cell). Methods
joined by a heavy bar are not separated at alpha=0.05 (Nemenyi post-hoc). When
the Friedman omnibus is itself not significant the generator renders the ranked
ladder descriptively only (faded, with a banner), so the figure cannot be
misread as a ranking - see experiment/_eval/_benchmark_visuals.render_cd_diagram.

Data source: the latest experiment/comparison_*.csv written by
run_aggregate_results.py, so the figure is pinned to one results snapshot
rather than the live experiment/ tree.

Usage: python fig_f1_critical_difference.py
"""
# Paper caption (for LaTeX; kept here so the how-to-read text survives the move
# off the figure image):
#   Critical-difference diagram (Demsar 2006, JMLR). Each method is ranked 1 (best)
#   to k within every full_features cell, then averaged over the n cells to give its
#   mean rank (dot, rightmost = best). Methods joined by a heavy horizontal bar
#   differ by less than the critical difference (Nemenyi post-hoc, alpha = 0.05) and
#   are therefore not significantly different. When the Friedman omnibus is itself
#   not significant the ranked ladder is shown faded with a banner: the order is
#   descriptive only and no pairwise difference is significant.
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
_sys.path.insert(0, str(_EXP / "_eval"))
from _parsing import latest_comparison_csv, entries_from_csv
from _benchmark_visuals import prepare_benchmark, render_cd_diagram

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
        if prep is None or prep["nem"] is None:
            print(f"  skip {tgt}: insufficient matched cells for Friedman/Nemenyi")
            continue
        mean_ranks, cd, p_value, n_cells, _k, _kept, _dropped = prep["nem"]
        out = HERE / "figures" / f"fig_f1_critical_difference_{tgt}"
        render_cd_diagram(
            mean_ranks, cd, prep["text_kept"], out,
            title=f"Critical Difference - {tgt} AUROC",
            p_value=p_value, n_cells=n_cells,
            arch_tuples=prep["arch_kept"], dropped_archs=prep["dropped_text"],
        )
        print("saved", out)


if __name__ == "__main__":
    main()
