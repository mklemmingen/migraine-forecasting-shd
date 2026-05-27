"""Figure F5 - icicle plot of the experiment leaves summarised in the
aggregate comparison CSV (Additions 0 / 1 + Addition 5 site externals).

One horizontal stripe per path level (addition / target / feature_set /
architecture / version / split ratio / split strategy / HP state / HP strategy /
HP operating point); each node's width is proportional to its descendant-leaf
count, and children sit under their parent so a reader can trace any leaf's full
configuration top-to-bottom [Shneiderman 1992; Andrews & Sanguinetti 2019].

Scope caveat: the comparison CSV (run_aggregate_results.py) is built from
the Addition 0 + Addition 1 grids plus the Addition 5 site externals. The
analytical-only Additions (Addition 2 explainability, Addition 3 temporal,
Addition 4 sequence, Addition 6 clinical value) write their outputs at the
addition root rather than as per-leaf scoring rows, so they do not appear
in the icicle. A reader who wants the full project decomposition (0 through
6) reads this figure together with the Addition-naming preamble of
results_findings.md.

Data source: the latest experiment/comparison_*.csv written by
run_aggregate_results.py - one row per leaf, so the icicle reflects exactly the
configurations summarised in that snapshot's comparison table. The CSV
timestamp is printed at runtime and is the load-bearing provenance for
the leaf count rendered in the figure.

Usage: python fig_f5_leaf_tree.py
"""
# Paper caption (for LaTeX; the layout description and citation were moved off the
# image to keep the title to one line):
#   Icicle plot of the discovered experiment leaves (Shneiderman 1992; Andrews &
#   Sanguinetti 2019). Each row is a path level (addition / target / feature set /
#   architecture / model version / split ratio / split strategy / HP-tuned? / HP
#   search strategy / HP operating point); a cell's width is proportional to its
#   descendant-leaf count and children sit directly under their parent, so any leaf's
#   full configuration reads top-to-bottom. Cells are coloured by entity (target,
#   feature set, architecture, split strategy) where a canonical colour exists;
#   purely structural levels are a neutral grey.
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
