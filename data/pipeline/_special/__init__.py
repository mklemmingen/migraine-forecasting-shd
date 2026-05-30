"""data.pipeline._special - modular special-case filters and sensitivity inputs.

Holds per-patient flags and cohort-subset definitions that are needed for
sensitivity analyses but are not part of the engineered-feature surface (e.g.
phenotype flags like migraine-with-aura status that the engineered features
deliberately omit because they are baseline characteristics, not daily diary
signals). Each module exposes a single extractor function that returns a
patient-keyed DataFrame; the corresponding persisted artefact lives under
``data/processed/special/`` so downstream analyses can load it without
re-touching the raw XLS.
"""
from .aura import extract_aura_status
from .cohort_metadata import extract_cohort_metadata

__all__ = ["extract_aura_status", "extract_cohort_metadata"]
