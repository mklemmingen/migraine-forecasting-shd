"""Driver for the personalisation + within-person evaluation layer (Addition 5).

Post-hoc pass over the selected Additions 0/1/4 leaves (like Addition 2's
insight pass): load each leaf's model.joblib, regenerate its test/val
predictions while RETAINING patient_id (the column the standard prep_split
drops), then:

  1. emit the standard results_*.txt for each personalisation regime under a
     parse_path-compatible path, so the regimes fold into the existing
     comparison_*.html alongside 0/1/4 (the comparability bridge);
  2. compute the within-person AUROC/AUPRC distribution and the meta-analytic
     within-person C-statistic per cell, and the pooled-vs-within gap (RQ2);
  3. emit the per-patient distribution + cold-start figures and a one-page
     personalisation comparison HTML.

Reads no new data beyond the existing split parquets and the saved leaf models.
Design and decisions: docs/addition5_personalization.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiment/5/
EXP = HERE.parent                               # experiment/
REPO = EXP.parent
sys.path[0:0] = [str(EXP), str(HERE), str(HERE / "_personal")]
from _dataRead.read import load_raw, prep_split  # noqa: E402
import within_person as WP   # noqa: E402
import regimes as RG         # noqa: E402
import walkforward as WF     # noqa: E402

TARGETS = ("headache", "migraine")
# Regimes scored into the comparison table; "pooled" reuses the existing leaf.
REGIMES = ("pooled", "per_patient", "partial_pool")


def regenerate_predictions(leaf_dir: Path, split_parquet: Path):
    """Load leaf_dir/model.joblib and predict on split_parquet, returning
    (y, p, patient_id) with patient_id retained for within-person grouping.

    TODO: load_raw the split (keeps patient_id+date), prep_split for the model's
    X, model.predict_proba, and return y/p aligned to the rows plus the
    patient_id column. Mirrors how Addition 2 regenerates predictions from the
    saved bundle.
    """
    raise NotImplementedError("prediction regeneration - see TODO")


def run_cell(target: str, feature_set: str, leaf_dir: Path) -> dict:
    """One (target, feature_set) cell across the regimes.

    TODO: for each regime in REGIMES, obtain (p_val, p_test); call
    RG.emit_holdout_results into experiment/5/<target>/<feature_set>/<regime>/
    <ratio>/<split>/ so it folds into comparison_*.html; then WP.per_patient_scores
    + WP.within_person_cstatistic on the test predictions, and WP.pooled_vs_within
    for RQ2. Collect the per-patient tables for the distribution figure and call
    WF.cold_start_curve for the cold-start figure. ~30-40 lines.
    """
    raise NotImplementedError("per-cell orchestration - see TODO")


def main():
    # TODO: discover the headline leaves per (target, feature_set) by reusing the
    # Addition 2 selection (experiment/2/select.py), run run_cell for each, and
    # write the per-patient distribution / cold-start figures and the
    # personalisation comparison HTML. The regime results_*.txt files are already
    # parse_path-compatible, so run_aggregate_results.py will fold them into the
    # main comparison table on its next run.
    raise NotImplementedError("driver orchestration - see TODO")


if __name__ == "__main__":
    main()
