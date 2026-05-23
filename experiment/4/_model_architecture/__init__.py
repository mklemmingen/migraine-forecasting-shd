"""Sequence-model architectures (Addition 4).

Each subdirectory is one architecture and exposes a single builder in its
``model.py``. The scaffold (experiment/4/_scaffold_leaves.py) registers each
via the ``VERSIONS`` tuple, exactly as Addition 1 does for the TabPFN family.

Builder contract
----------------
Every ``build_<arch>(X_train, y_train, *, device='cuda', random_state=0,
output_dir=None)`` must:

  - take an id-bearing ``X_train`` (pandas.DataFrame still carrying
    ``patient_id`` and ``date`` - loaded via _seq.dataread.load_seq, NOT the
    Additions 0/1 prep_split) and target ``y_train`` (1-D {0,1}).
  - accept ``device`` and ``random_state`` as keyword-only arguments.
  - return a FITTED estimator with ``.predict_proba(X) -> ndarray (n, 2)``
    that is row-aligned to X (one probability per input row), so the leaf
    evaluate.py is identical to Additions 0/1. The estimator is the shared
    _seq.sklearn_wrapper.SequenceClassifier, parameterised by this
    architecture's nn.Module factory.
  - leave imbalance handling to the wrapper's weighted objective and the
    external threshold-selection step in evaluate.py (no resampling, so the
    calibration panel stays interpretable).
  - move the model to CPU before returning if ``output_dir`` serialisation
    portability is required (train.py joblib-dumps the returned object).

Adding a new architecture
-------------------------
1. Create ``_model_architecture/<arch>/__init__.py`` (empty) and ``model.py``
   with the nn.Module factory + ``build_<arch>``.
2. Add a ``Version("<label>", "<arch>", "build_<arch>")`` entry to
   ``VERSIONS`` in _scaffold_leaves.py.
3. Run ``python experiment/4/_scaffold_leaves.py --force`` to regenerate the
   leaf scripts.
"""
