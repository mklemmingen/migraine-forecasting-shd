"""Sequence-windowing layer for Addition 4 (experiment/4).

Three modules, imported by the self-windowing sklearn estimator and the
leaf templates:

  windowing.py       - gap-aware, within-split, per-patient look-back windows
  sklearn_wrapper.py - SequenceClassifier: sklearn-style fit/predict_proba
                       that builds windows internally and returns row-aligned
                       probabilities, so the leaf evaluate.py stays identical
                       to Additions 0/1
  dataread.py        - load_seq: like _dataRead.read but keeps patient_id and
                       date on X (the columns the windower needs)

Design rationale and literature anchors are in docs/addition4_sequence.md.
"""
