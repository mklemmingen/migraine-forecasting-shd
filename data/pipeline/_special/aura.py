"""aura.py - Extract migraine-with-aura status from the Park 2016 patient-baseline sheet.

The Park 2016 cohort comprises 60 patients with migraine without aura and 2 patients
with migraine with aura per ICHD-3 beta classification [park2016shd, p. 5, Tab. 1].
Aura status is a baseline clinical characteristic, not a daily diary signal, so it
is intentionally absent from the engineered-feature surface. This module reads the
patient-baseline sheet (Sheet 0, "62patients") of the raw XLS, finds the migraine
diagnosis column (Korean: 편두통 진단), and maps each patient's registration code
(등록번호; e.g. ``cmc-0026``) to an aura flag.

The output DataFrame has one row per patient with columns:

- ``patient_id`` (str): uppercased registration code matching the processed parquet
  format (e.g. ``CMC-0026``, ``DHA-0057``)
- ``has_aura`` (bool): True iff the patient has migraine with aura (조짐편두통)

Use the persisted artefact at ``data/processed/special/aura_status.parquet`` rather
than re-reading the raw XLS in downstream analyses.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# Sheet 0 has a single header row (the Korean column names) sitting at row index
# 1; row 0 is the publication-title row "S1 File. Dataset of 62 patients" and
# data starts at row 2. Using ``header=1`` lands the 62 patient rows correctly;
# an earlier ``header=[1, 2]`` two-row read silently consumed CMC-0001's data row
# as a stale second header, dropping the first patient and yielding 61 rows.
_PID_NAME = "고유번호"             # numeric patient unique number
_REG_NAME = "등록번호"             # registration code 'cmc-XXXX' / 'dha-XXXX'
_DX_NAME  = "편두통 진단"          # migraine diagnosis

# Aura status is encoded as a Korean substring in the diagnosis text:
#   '조짐편두통'    -> migraine with aura (조짐 = aura)
#   '무조짐편두통' -> migraine without aura (무조짐 = without aura)
# Substring matching needs the negative check on '무조짐' to avoid the
# without-aura cases matching '조짐'.
_AURA_TOKEN     = "조짐"
_NO_AURA_TOKEN  = "무조짐"


def extract_aura_status(raw_xls_path: str | Path) -> pd.DataFrame:
    """Return patient-keyed aura status from the Sheet 0 baseline table of the
    raw Park 2016 XLS. Returns 62 rows: 60 patients with migraine without aura
    (무조짐편두통) and 2 with migraine with aura (조짐편두통, CMC-0026 and
    DHA-0057 in the canonical uppercased processed-parquet ID scheme)."""
    df = pd.read_excel(raw_xls_path, sheet_name=0, header=1)
    df = df[df[_PID_NAME].notna()].copy()

    dx = df[_DX_NAME].astype(str)
    has_aura = dx.str.contains(_AURA_TOKEN, na=False) & ~dx.str.contains(_NO_AURA_TOKEN, na=False)

    out = pd.DataFrame({
        "patient_id": df[_REG_NAME].astype(str).str.upper(),
        "has_aura": has_aura.values,
    }).reset_index(drop=True)

    # Defensive guards against the silent-row-drop failure mode that an earlier
    # ``header=[1, 2]`` two-row read introduced (it consumed CMC-0001's data as
    # a stale second header, yielding 61 rows + 59 no-aura instead of 62 + 60).
    # The Park 2016 enrolment is fixed at 62 patients, 60 without aura, 2 with
    # aura per the Methods dataset description and dataset.md §"Cohort"; any
    # deviation here means the raw read shifted under us and the downstream
    # sensitivity will be off.
    assert len(out) == 62, f"Expected 62 patient rows, got {len(out)}"
    assert out["has_aura"].sum() == 2, f"Expected 2 aura patients, got {out['has_aura'].sum()}"
    assert (~out["has_aura"]).sum() == 60, f"Expected 60 no-aura patients, got {(~out['has_aura']).sum()}"
    return out
