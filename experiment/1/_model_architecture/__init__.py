"""TabPFN-family model architectures (Addition 1).

Each subdirectory is one architecture variant and exposes a single builder
function in its ``model.py``. The scaffold (``experiment/1/_scaffold_leaves.py``)
registers each variant via the ``VERSIONS`` tuple - see that file's docstring.

Builder contract
----------------
Every ``build_<variant>(X_train, y_train, *, device='cuda', random_state=0)``
must:

  - take training features ``X_train`` (pandas.DataFrame or 2-D ndarray) and
    target ``y_train`` (1-D ndarray of {0, 1}) - no separate calibration
    set; the calibration-policy rationale is documented in
    ``docs/tabPfn.MD`` §4.
  - accept ``device`` and ``random_state`` as keyword-only arguments
    (the ``*`` in the signature enforces this).
  - return a fitted estimator with ``.predict_proba(X) -> ndarray of
    shape (n, 2)``, sklearn-compatible.
  - leave imbalance handling to the external threshold-selection step in
    each leaf's ``evaluate.py`` / ``evaluate_cv.py``.

Adding a new variant
--------------------
1. Create ``_model_architecture/<variant>/__init__.py`` (empty) and
   ``model.py`` with the builder.
2. Add a ``Version("<label>", "<variant>", "build_<variant>")`` entry to
   ``VERSIONS`` in ``_scaffold_leaves.py``.
3. Run ``python experiment/1/_scaffold_leaves.py --force`` to regenerate
   the leaf scripts - every (target × feature_set × ratio × split_type)
   combination gets its own train/evaluate/(evaluate_cv) trio under
   ``version_<label>/``.
"""
