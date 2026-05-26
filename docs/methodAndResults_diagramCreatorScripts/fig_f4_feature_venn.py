"""Figure F4 - feature-set inclusion Venn diagrams.

Two panels over the three nested-ish engineered sets (full, spano, no_rolling):
a region-count Venn and a names Venn that lists every feature coloured by origin
(original SHD column vs engineered rolling/lag/interaction/state feature). The
Park set is shown as a sidebar because it is a small, largely disjoint clinical
subset rather than a Venn region.

Unlike the other figures this one cannot read the comparison CSV: feature
*names* are not a CSV column. It recomputes the sets directly from the canonical
diary_train.parquet by running the same filter modules the experiments use, so
the membership is the ground truth rather than a transcription.

Usage: python fig_f4_feature_venn.py
"""
# Paper caption (for LaTeX):
#   Feature-set inclusion as a nested Euler diagram. spano and no_rolling are strict
#   subsets of full (no feature lies outside full), so each is drawn wholly inside the
#   full circle, overlapping each other; a three-circle Venn would falsely show them
#   bulging outside full. The layout is schematic - circle areas are not to scale; the
#   exact sizes are the region-count labels (counts variant), and the names variant
#   lists every feature per region coloured by origin (original SHD column vs
#   engineered: rolling / lag / interaction / state-derived). Park is a near-subset
#   (5 of 6 features in full, plus one derived hormonal-changes flag), shown as a
#   sidebar rather than a fourth circle.
import sys as _sys
from pathlib import Path as _Path
_EXP = _Path(__file__).resolve().parents[2] / "experiment"
_sys.path.insert(0, str(_EXP))
_sys.path.insert(0, str(_EXP / "_eval"))
from _venn_diagrams import (
    compute_feature_sets, generate_count_venn_png, generate_names_venn_png,
)

HERE = _Path(__file__).resolve().parent


def main():
    feature_sets = compute_feature_sets(_EXP)
    if feature_sets is None:
        raise SystemExit("no processed diary_train.parquet found - run the data pipeline first")
    sizes = {k: len(v) for k, v in feature_sets.items()}
    print(f"  feature-set sizes {sizes}")
    counts_out = HERE / "figures" / "fig_f4_feature_venn_counts"
    names_out  = HERE / "figures" / "fig_f4_feature_venn_names"
    generate_count_venn_png(feature_sets, counts_out)
    print("saved", counts_out)
    generate_names_venn_png(feature_sets, names_out)
    print("saved", names_out)


if __name__ == "__main__":
    main()
