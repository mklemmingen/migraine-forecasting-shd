"""Shared training-side helpers used by every leaf's train.py and the
CV evaluator's per-fold refit. Currently exposes the per-leaf stdout /
stderr capture context manager; see ``_training_script_output``."""
