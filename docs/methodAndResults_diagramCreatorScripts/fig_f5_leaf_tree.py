"""Figure F5 - icicle plot of the discovered experiment leaves.

One horizontal stripe per path level (addition / target / feature_set /
architecture / version / split ratio / split strategy / HP state / HP strategy /
HP operating point); each node's width is proportional to its descendant-leaf
count, and children sit under their parent so a reader can trace any leaf's full
configuration top-to-bottom [Shneiderman 1992; Andrews & Sanguinetti 2019].

Data source: the latest experiment/comparison_*.csv written by
run_aggregate_results.py - one row per leaf, so the icicle reflects exactly the
configurations summarised in that snapshot's comparison table.

Usage: python fig_f5_leaf_tree.py
"""
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
_sys.path.insert(0, str(_EXP / "_eval"))
from _parsing import latest_comparison_csv, entries_from_csv
from _tree_diagram import generate_tree_png

HERE = _Path(__file__).resolve().parent


def main():
    csv_path = latest_comparison_csv(_EXP)
    if csv_path is None:
        raise SystemExit("no comparison_*.csv in experiment/ - run run_aggregate_results.py first")
    entries = entries_from_csv(csv_path)
    print(f"  source {csv_path.name} ({len(entries)} leaves)")
    out = HERE / "figures" / "fig_f5_leaf_tree"
    generate_tree_png(entries, out)
    print("saved", out)


if __name__ == "__main__":
    main()
