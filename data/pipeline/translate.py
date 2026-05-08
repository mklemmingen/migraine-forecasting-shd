"""
translate.py — Step 1: Korean → English translation of Sheet 3 (total diary 4579).

Sheet 3 structure:
  105 columns, 4,591 rows (4,579 diary days + title row + 2-row header + 1 absorbed totals row).
  Data begins at Row 3. Includes every diary day — headache and non-headache — for all 62 patients.

Outcome column polarity: `두통이없는날` means "headache-free day" — Y = no headache.
This is inverted during translation so headache_free=1 means the patient was headache-free.
The polarity is then corrected to migraine_today at the engineering step.

Known data quality issue: trigger factor columns contain a spurious row where each value equals
the column-wide count of positives. This totals row is excluded by filtering on valid patient IDs.
"""
import numpy as np
import pandas as pd


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill and flatten MultiIndex columns into '{group} — {subheader}' format."""
    if isinstance(df.columns, pd.MultiIndex):
        level_0 = (pd.Series(df.columns.get_level_values(0))
                   .replace(r'^Unnamed:.*', np.nan, regex=True)
                   .ffill())
        level_1 = df.columns.get_level_values(1)
        new_cols = []
        for g, s in zip(level_0, level_1):
            if pd.isna(g) or g == s or str(s).startswith('Unnamed:'):
                new_cols.append(str(g) if pd.notna(g) else str(s))
            else:
                new_cols.append(f"{g} — {s}")
        df.columns = new_cols
    return df


def translate_sheet3(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Step 1 — Translate Sheet 3 from Korean to English.

    Applies the verbatim Korean → English column mapping, standardises types,
    and removes the absorbed totals row. Returns a clean DataFrame with no
    'split' column — splitting is handled by the splits sub-package.

    Column selection rationale (Park et al. 2016):
    - All 18 Park et al. trigger factors retained for model-driven selection.
      Low-prevalence triggers (sunlight 0.8%, cheese_chocolate 0.7%,
      inappropriate_lighting 0.2%, exercise_as_trigger 1.3%, excessive_smoking)
      are included so that SHAP (Addition 2) can assess their contribution
      empirically rather than pre-filtering on same-day p-values from a
      different analytical question.
    - other_trigger excluded: unstructured catch-all, not a validated instrument item.
    - weather column read from Korean header to bypass Spano 2026 pipeline typo that zeroed 226 rows
    """
    df = raw_df.copy()
    df = _flatten_multiindex_columns(df)

    col_map = {
        '번호': 'entry_id',
        '등록번호': 'patient_id',
        '날짜': 'date',
        '두통이없는날': 'headache_free',
        '두통지속중여부': 'headache_ongoing',
        '예방약 사용': 'preventive_medication',
        '정도': 'severity_category',
        'Pain intensity (VAS)': 'severity_vas',
        '내인적 요인 — 스트레스': 'stress',
        '내인적 요인 — 수면과다': 'oversleeping',
        '내인적 요인 — 수면부족': 'lack_of_sleep',
        '내인적 요인 — 운동 안하기': 'no_exercise',
        '내인적 요인 — 육체적 피로': 'physical_fatigue',
        '내인적 요인 — 생리주기 : 월경기': 'menstruation',
        '내인적 요인 — 생리주기 : 배란기': 'ovulation',
        '내인적 요인 — 과도한 감정변화': 'emotional_changes',
        '외부적 요인 — 날씨/온도 변화': 'weather_change',
        '외부적 요인 — 소음': 'noise',
        '외부적 요인 — 특정한 냄새(화장품 향수 등)': 'specific_smells',
        '기타 — 과도한 음주': 'alcohol',
        '기타 — 불규칙한 식사(공복 등)': 'irregular_meals',
        '기타 — 과식': 'overeating',
        '기타 — 과도한 카페인 음료': 'excessive_caffeine',
        '기타 — 여행': 'travel',
        '격렬한 운동(분)': 'vigorous_exercise_min',
        '중등도운동(분)': 'moderate_exercise_min',
        '내인적 요인 — 운동': 'exercise_as_trigger',  # p=0.78 in Park; 1.3% prevalence
        '외부적 요인 — 과도한 햇빛': 'sunlight',  # p=0.73; 0.8% prevalence
        '외부적 요인 — 부적절한 조명': 'inappropriate_lighting',  # 0.2% prevalence
        '기타 — 과도한 흡연': 'excessive_smoking',  # p=0.73
        '기타 — 치즈 초콜릿': 'cheese_chocolate',  # 0.7% prevalence
    }

    keep_cols = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(keep_cols.keys())].rename(columns=keep_cols)

    df = df.dropna(subset=['patient_id'])
    df = df[~df['patient_id'].astype(str).str.contains('합계|total', case=False, na=False)]

    df['patient_id'] = df['patient_id'].astype(str).str.upper().str.strip()
    df['date'] = pd.to_datetime(
        df['date'].astype(str).str.extract(r'(\d{8})')[0], format='%Y%m%d')
    df['headache_free'] = df['headache_free'].apply(
        lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

    for col in ['vigorous_exercise_min', 'moderate_exercise_min']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    trigger_cols = [v for k, v in col_map.items() if v not in (
        'entry_id', 'patient_id', 'date', 'headache_free', 'headache_ongoing',
        'preventive_medication', 'severity_category', 'severity_vas',
        'vigorous_exercise_min', 'moderate_exercise_min',
    )]
    for col in trigger_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df
