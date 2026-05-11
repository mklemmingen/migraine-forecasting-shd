from pathlib import Path

from tabpfn_extensions.post_hoc_ensembles import AutoTabPFNClassifier

# max_time covers ~20 configs × 8 bag folds = 160 sub-fits on this GPU,
# where each TabPFN fit costs ~13 s. See docs/tabPfn.MD §5.2 for the
# budget calibration.
_MAX_TIME_SECONDS = 7200


def build_autotabpfn(X_train, y_train, *, device='cuda', random_state=0, output_dir=None):
    """AutoTabPFN - post-hoc ensemble of TabPFN configurations stacked
    via AutoGluon as the meta-learner.

    The AutoGluon TabularPredictor inside AutoTabPFNClassifier serialises
    to its own on-disk directory; the joblib pickle of the wrapper holds
    only a path reference to that directory. To keep the AutoGluon
    predictor co-located with the leaf's `model.joblib`, the leaf's
    train.py passes its EXPERIMENT_DIR as `output_dir`, which this
    builder forwards to AutoGluon via `phe_init_args={'path': ...}`. Any
    non-AutoTabPFN builder ignores `output_dir` - see the contract in
    `_model_architecture/__init__.py`.

    Per-leaf training-time stdout/stderr (including AutoGluon's
    leaderboard, ensemble weights, time-limit warnings, etc.) is
    captured by the generic ``_eval._training_script_output``
    context manager wrapping each leaf's ``train.py`` ``main()``.

    No dedicated peer-reviewed paper documents AutoTabPFN as a named
    classifier; the methodology section must disclose this rather than
    imply otherwise. See docs/tabPfn.MD §5.2 for citation framing.
    """
    phe_init_args = None
    if output_dir is not None:
        phe_init_args = {'path': str(Path(output_dir) / 'autogluon_predictor')}
    base = AutoTabPFNClassifier(
        max_time=_MAX_TIME_SECONDS,
        device=device,
        random_state=random_state,
        phe_init_args=phe_init_args,
    )
    base.fit(X_train, y_train)
    return base
