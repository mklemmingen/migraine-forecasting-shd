"""experiment._eval._special - sensitivity analyses that exclude or condition on
patient-level phenotype subsets defined under ``data/pipeline/_special`` and
persisted to ``data/processed/special``. Each script here is a one-shot runner
that loads a saved phenotype-flag artefact, recomputes headline metrics on the
filtered subset of canonical headline leaves, and emits a summary CSV alongside
the standard contract for downstream comparison.
"""
