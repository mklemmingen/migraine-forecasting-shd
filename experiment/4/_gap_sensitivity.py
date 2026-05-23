"""Gap-threshold sensitivity sweep for Addition 4 (Decision 2,
docs/addition4_sequence.md Section 9).

Varies ``gap_segment_days`` over {3, 7, 14} on the headline
migraine/full_features chronological cell and reports the PRIMARY estimate
(5-fold expanding-window TimeSeriesSplit AUROC, mean +/- std) per architecture.
The question is robustness, not tuning: does the sequence-vs-tabular conclusion
move with the gap threshold? Fixed seed; CPU.

Run: CUDA_VISIBLE_DEVICES="" .venv/bin/python experiment/4/_gap_sensitivity.py
"""
import datetime as _dt
import sys
import uuid
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent          # experiment/4/
EXP = HERE.parent                               # experiment/
REPO = EXP.parent
sys.path[0:0] = [str(EXP), str(HERE), str(HERE / "_seq")]
from _dataRead.read import load_raw, chronological_subsplit  # noqa: E402
from _seq.dataread import seq_split  # noqa: E402
from sklearn_wrapper import SequenceClassifier  # noqa: E402
from _model_architecture.gru.model import _GRUNet  # noqa: E402
from _model_architecture.tcn.model import _TCNNet  # noqa: E402
from _model_architecture.window_mlp.model import _WindowMLPNet  # noqa: E402

GAPS = (3, 7, 14)
# label -> (module factory, hidden, dropout) matching the build_* defaults
ARCHS = {
    "window-mlp": (_WindowMLPNet, 32, 0.4),
    "gru":        (_GRUNet, 16, 0.3),
    "tcn":        (_TCNNet, 16, 0.3),
}
N_SPLITS = 5
CAL_RATIO = 0.20
TARGET = "migraine"
FEATURE_SET = "full_features"


def cv_auroc(cv, factory, hidden, dropout, gap):
    """5-fold expanding-window TimeSeriesSplit AUROC for one (arch, gap)."""
    aurocs = []
    for fold in range(1, N_SPLITS + 1):
        train_fold = cv[cv["cv_fold"] < fold].copy()
        val_fold = cv[cv["cv_fold"] == fold].copy()
        if train_fold.empty:
            continue
        tr, _cal = chronological_subsplit(train_fold, cal_ratio=CAL_RATIO)
        X_tr, y_tr = seq_split(tr)
        X_v, y_v = seq_split(val_fold)
        if len(np.unique(y_v)) < 2:
            continue
        model = SequenceClassifier(
            factory, lookback=7, gap_segment_days=gap, hidden=hidden,
            dropout=dropout, weight_decay=1e-3, lr=1e-3, epochs=100,
            device="cpu", random_state=42,
        )
        model.fit(X_tr, y_tr)
        p_v = model.predict_proba(X_v)[:, 1]
        aurocs.append(roc_auc_score(y_v.values, p_v))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def main():
    cv = load_raw(str(REPO / "data" / "processed" / TARGET / "diary_cv5_timeseries.parquet"))
    header = (f"Gap-threshold sensitivity | {TARGET}/{FEATURE_SET} | "
              f"5-fold TimeSeriesSplit AUROC (mean +/- std)")
    col = "arch         | " + " | ".join(f"gap={g:<2}    " for g in GAPS)
    lines = [header, "=" * len(col), col, "-" * len(col)]
    print("\n".join(lines))
    for label, (factory, hidden, dropout) in ARCHS.items():
        cells = []
        for g in GAPS:
            mean, std = cv_auroc(cv, factory, hidden, dropout, g)
            cells.append(f"{mean:.3f}+-{std:.3f}")
        row = f"{label:<12} | " + " | ".join(f"{c:<11}" for c in cells)
        print(row)
        lines.append(row)
    lines.append("=" * len(col))

    out = HERE / f"gap_sensitivity_{_dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
