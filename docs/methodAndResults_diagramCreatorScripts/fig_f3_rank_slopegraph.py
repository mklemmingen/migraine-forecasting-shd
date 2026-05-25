"""Figure F3 - rank slopegraphs for AUROC [Tufte 2001].

Per-architecture rank trajectory across the full_features cells, per target.
A flat line is a stable architecture (same rank everywhere); crossing lines
mark architectures whose standing swings with the (feature_set, ratio, split)
cell. Same coverage-filtered architecture set as the critical-difference and
performance-profile views.

Data source: the latest experiment/comparison_*.csv written by
run_aggregate_results.py.

Usage: python fig_f3_rank_slopegraph.py
"""
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
_sys.path.insert(0, str(_EXP / "_eval"))
from _parsing import latest_comparison_csv, entries_from_csv
from _benchmark_visuals import prepare_benchmark, render_rank_slopegraph

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
        out = HERE / "figures" / f"fig_f3_rank_slopegraph_{tgt}"
        render_rank_slopegraph(
            prep["matrix_kept"], prep["cell_labels"], prep["text_kept"], out,
            title=f"Rank slopegraph - {tgt} AUROC",
            higher_is_better=prep["higher_is_better"],
            arch_tuples=prep["arch_kept"], dropped_archs=prep["dropped_text"],
        )
        print("saved", out)


if __name__ == "__main__":
    main()
