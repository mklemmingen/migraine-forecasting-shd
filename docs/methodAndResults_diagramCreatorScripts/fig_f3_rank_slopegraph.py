"""Figure F3 - rank slopegraphs for AUROC [Tufte 2001], within-family scope.

Per-architecture rank trajectory across the full_features cells, per target.
A flat line is a stable architecture (same rank everywhere); crossing lines
mark architectures whose standing swings with the (feature_set, ratio, split)
cell. Same coverage-filtered architecture set as the critical-difference and
performance-profile views - that filter restricts the comparison to HP
variants of stacked_2xgb against TabPFN single-fit versions; cross-family
entries (add0_stacked_NonHP, add4_window_mlp, pooled_lr) drop out for
incomplete coverage, which makes f3 a WITHIN-FAMILY rank trajectory rather
than a cross-architecture rank test.

Data source: the latest experiment/comparison_*.csv written by
run_aggregate_results.py.

Usage: python fig_f3_rank_slopegraph.py
"""
# §11 compliance: Tufte rank slopegraph with omnibus Friedman p (no per-method CI applicable).
#   §11.1 N/A (rank metric); §11.3 caption: cohort+n in rendered title
#   §11.6, §11.7 via renderer helper; §11.10 no banned adjectives; §11.11 self-check
# Paper caption (for LaTeX):
#   Rank slopegraph (Tufte 2001). Each polyline is one architecture's mean-rank
#   trajectory across the full_features cells (1 = best). Flat lines are stable
#   architectures; crossing lines swap rank between cells. A rank orders per-cell
#   point estimates, so a crossing smaller than the critical difference (see the CD
#   diagram) is not a significant difference; when the Friedman omnibus is not
#   significant the whole ladder is faded and banner-flagged as descriptive only.
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
        # Carry the Friedman omnibus significance so the slopegraph fades and
        # banners a non-significant ordering, exactly as the CD diagram does.
        nem = prep["nem"]
        p_value = nem[2] if nem else None
        omnibus_sig = (p_value is None) or (p_value < 0.05)
        out = HERE / "figures" / f"fig_f3_rank_slopegraph_{tgt}"
        render_rank_slopegraph(
            prep["matrix_kept"], prep["cell_labels"], prep["text_kept"], out,
            title=f"Rank slopegraph - {tgt} AUROC",
            higher_is_better=prep["higher_is_better"],
            arch_tuples=prep["arch_kept"], dropped_archs=prep["dropped_text"],
            omnibus_sig=omnibus_sig, p_value=p_value,
        )
        print("saved", out)


if __name__ == "__main__":
    main()
