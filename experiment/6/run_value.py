"""Driver for the clinical-forecast-value layer (Addition 6).

Post-hoc pass (like Addition 2): load the selected leaf models from Additions
0/1/4/5, regenerate their calibrated test predictions, and add the value layer:

  1. decision curves (net benefit vs threshold) overlaying the architectures per
     (target, feature_set) cell - so the value comparison spans additions;
  2. Brier skill vs each patient's training-set climatology;
  3. the operating-point mapping (val-MCC threshold vs net-benefit-optimal) and
     sensitivity at a tolerated false-alarm rate;
  4. (optional, headache only) the measured-weather ablation.

Adds no core model and does not touch evaluate.py. No clinical-utility claim is
made beyond the development stage [vasey2022decideAI, p. 1].
Design and decisions: docs/addition6_clinical_value.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiment/6/
EXP = HERE.parent                               # experiment/
REPO = EXP.parent
sys.path[0:0] = [str(EXP), str(HERE), str(HERE / "_value")]
from _dataRead.read import load_raw, prep_split  # noqa: E402
import decision_curve as DC   # noqa: E402
import skill as SK            # noqa: E402
import operating_point as OP  # noqa: E402

TARGETS = ("headache", "migraine")


def cell_value(leaf_models, val_split, test_split, train_split) -> dict:
    """Value layer for one (target, feature_set) cell across architectures.

    TODO: for each architecture's leaf model, regenerate calibrated p_test (and
    p_val), then:
      - DC.decision_curve(y_test, p_test) -> store the model curve for the
        cross-architecture overlay;
      - climatology from the TRAIN split base rates (SK.per_patient_climatology),
        then SK.brier_skill_score(y_test, p_test, ref);
      - OP.map_threshold(val_mcc_threshold, curve thresholds, model NB) and
        OP.sensitivity_at_fpr(y_test, p_test).
    The decision-curve primitives are already implemented; this function is the
    orchestration + per-row prediction regeneration (retaining patient_id for the
    climatology). ~30-40 lines.
    """
    raise NotImplementedError("per-cell value orchestration - see TODO")


def main():
    # TODO: discover the leaves to compare (reuse experiment/2/select.py across
    # Additions 0/1/4/5), run cell_value per (target, feature_set), and emit:
    #   - one decision-curve figure per cell overlaying all architectures
    #     (cross-addition by construction),
    #   - a Brier-skill-vs-climatology table,
    #   - comparison_value_<ts>.html.
    # Optionally run the headache-only measured-weather ablation
    # (_value/weather_join.py) if the external data is sourced.
    raise NotImplementedError("driver orchestration - see TODO")


if __name__ == "__main__":
    main()
