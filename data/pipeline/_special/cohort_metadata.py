"""cohort_metadata.py - Extract patient-level site + sex metadata from the
Park 2016 baseline sheet.

Like ``_special/aura.py``, these patient-level attributes are baseline
demographic / recruitment characteristics absent from the engineered diary-day
feature surface but needed for sensitivity analyses (e.g. stratum
disaggregation of the within-person C-statistic per the Pierson / equity ask
at body §3.6, the panel-revision T3-6 work). Reading sheet 0 of the raw XLS
with ``header=1`` recovers 62 patient rows; the same header bug discipline as
``_special/aura.py`` applies and a defensive assert guards against silent row
drops if the XLS structure shifts.

Output columns:

- ``patient_id`` (str): uppercased registration code ('CMC-0001', 'DHA-0057')
- ``site`` (str): 'Uijeongbu' or 'Dongtan' (translated from the Korean
  recruitment hospital names 의정부 / 동탄 한림)
- ``sex`` (str): 'female' or 'male' (translated from 여성 / 남성)

Use the persisted artefact at ``data/processed/special/cohort_metadata.parquet``
rather than re-reading the raw XLS in downstream analyses.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


_PID_NAME = "고유번호"             # numeric patient unique number
_REG_NAME = "등록번호"             # registration code 'cmc-XXXX' / 'dha-XXXX'
_SEX_NAME = "성별"                 # sex column (여성=female, 남성=male)
_SITE_NAME = "병원"                # hospital column (의정부=Uijeongbu, 동탄 한림=Dongtan)

# Korean → English translations for the two attribute domains. Map both the
# canonical site strings AND a partial-match prefix so 'Uijeongbu St. Mary's'
# (body §2.1) and '동탄 한림' (raw sheet) both resolve cleanly.
_SEX_MAP = {"여성": "female", "남성": "male"}
_SITE_MAP = {"의정부": "Uijeongbu", "동탄 한림": "Dongtan"}


def extract_cohort_metadata(raw_xls_path: str | Path) -> pd.DataFrame:
    """Return patient-keyed site + sex metadata from Sheet 0 of the raw Park
    2016 XLS. Returns 62 rows matching body §2.1 (51 female + 11 male; 32
    Uijeongbu + 30 Dongtan). patient_id is uppercased to match the canonical
    processed-parquet ID scheme so downstream joins work without re-mapping.
    """
    df = pd.read_excel(raw_xls_path, sheet_name=0, header=1)
    df = df[df[_PID_NAME].notna()].copy()

    sex = df[_SEX_NAME].map(_SEX_MAP)
    site = df[_SITE_NAME].map(_SITE_MAP)

    out = pd.DataFrame({
        "patient_id": df[_REG_NAME].astype(str).str.upper(),
        "site": site.values,
        "sex": sex.values,
    }).reset_index(drop=True)

    # Defensive guards against silent header drift in the raw XLS; the cohort
    # composition is fixed at 62 patients, 51 / 11 sex split, 32 / 30 site
    # split per Park 2016 SHD enrolment (body §2.1, docs/dataset.md cohort
    # table). Any deviation here means the raw read shifted under us.
    assert len(out) == 62, f"Expected 62 patient rows, got {len(out)}"
    assert (out["sex"] == "female").sum() == 51, \
        f"Expected 51 female patients, got {(out['sex'] == 'female').sum()}"
    assert (out["sex"] == "male").sum() == 11, \
        f"Expected 11 male patients, got {(out['sex'] == 'male').sum()}"
    assert (out["site"] == "Uijeongbu").sum() == 32, \
        f"Expected 32 Uijeongbu patients, got {(out['site'] == 'Uijeongbu').sum()}"
    assert (out["site"] == "Dongtan").sum() == 30, \
        f"Expected 30 Dongtan patients, got {(out['site'] == 'Dongtan').sum()}"
    return out
