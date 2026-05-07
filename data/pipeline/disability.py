"""
disability.py — Step 3: Sheet 2 (headache diary 1099) disability supplement.

Sheet 2 is headache-event-level: only rows where a headache occurred.
It is processed separately and joined to the daily diary by (patient_id, date)
for Stage 5 experiments only.

The returned DataFrame has no 'split' column. The caller must apply a split
using data/pipeline/splits/*.apply_split(..., boundaries=<diary_boundaries>)
so that the disability split is aligned with the diary split.
"""
import pandas as pd
from .translate import _flatten_multiindex_columns


def process_disability_sheet(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Translate Sheet 2 disability data from Korean to English.

    Returns a clean DataFrame with no 'split' column.
    """
    df = raw_df.copy()
    df = _flatten_multiindex_columns(df)

    col_map = {
        '번호': 'entry_id',
        '등록번호': 'patient_id',
        '날짜': 'date',
        '정도': 'severity_category',
        'Pain intensity (VAS)': 'severity_vas',
        '동반증상 — 편두통유무': 'migraine_flag',
        '유발요인 — 유발요인 수': 'trigger_count',
        '지속시간(분)': 'headache_duration_min',
        '구급약 사용1 — 장애': 'disability_any',
        '학교 또는 직장 — 결근하거나 등교하지 못하였다': 'disability_work_severe',
        '학교 또는 직장 — 작업또는 학업능률이 절반 이하로 감소하였다': 'disability_work_moderate',
        '학교 또는 직장 — 합계': 'disability_work_affected',
        '집안에서 — 가사일을 전혀 할 수 없었다': 'disability_housework_severe',
        '집안에서 — 가사의 능률이 절반 이하로 감소하였다': 'disability_housework_moderate',
        '집안에서 — 합계': 'disability_housework_affected',
        '모임/여가 활동 — 예정이 있었으나 참여 할 수 없었다': 'disability_social',
        # 18 trigger flags — same names as translated.parquet for join compatibility
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
    }

    keep_cols = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(keep_cols.keys())].rename(columns=keep_cols)

    df = df.dropna(subset=['patient_id'])
    df = df[~df['patient_id'].astype(str).str.contains('합계|total', case=False, na=False)]

    df['patient_id'] = df['patient_id'].astype(str).str.upper().str.strip()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(
            df['date'].astype(str).str.extract(r'(\d{8})')[0], format='%Y%m%d')

    return df
