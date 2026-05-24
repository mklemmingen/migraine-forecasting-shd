"""Hospital-site join for the leave-one-site-out external validation (Route A).

The SHD cohort was recruited at two Korean university hospitals; the site is
patient-level in Sheet 1 of the source workbook (not in the engineered diary),
so this utility recovers it (`등록번호` registration id -> `병원` hospital) and
attaches it to any frame keyed by patient_id. The two sites differ in base rate
(Uijeongbu ~8.6% vs Dongtan ~5.7% migraine days), which is what makes a site
split a genuine transportability test. See docs/external_validation_site.md.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SHEET1 = REPO / "data" / "raw" / "SHD-Dataset.xls"
SITES = ("uijeongbu", "dongtan")
_LABEL = {"의정부": "uijeongbu", "동탄 한림": "dongtan"}


@lru_cache(maxsize=1)
def patient_site_map() -> dict:
    """{uppercased registration id -> ascii site slug} from Sheet 1."""
    s1 = pd.read_excel(SHEET1, sheet_name="62patients", header=1)
    reg = next(c for c in s1.columns if "등록번호" in str(c))
    hosp = next(c for c in s1.columns if "병원" in str(c))
    sub = s1[[reg, hosp]].dropna()
    sub[reg] = sub[reg].astype(str).str.upper().str.strip()
    sub = sub[sub[reg].str.len() > 2]
    return {r: _LABEL.get(str(h).strip(), str(h).strip()) for r, h in zip(sub[reg], sub[hosp])}


def attach_site(df: pd.DataFrame, patient_col: str = "patient_id") -> pd.DataFrame:
    """Return df with a ``site`` column (NaN for the unmapped reconciliation
    patient); call ``.dropna(subset=['site'])`` before splitting."""
    m = patient_site_map()
    out = df.copy()
    out["site"] = out[patient_col].astype(str).str.upper().map(m)
    return out
