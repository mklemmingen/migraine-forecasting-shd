"""Extract per-leaf AutoGluon wall-clock from the on-disk training logs.

AutoTabPFN (`version_2-5-auto`) emits the line
    `AutoGluon training complete, total runtime = NNN.NNs ...`
inside `<leaf>/_running_output/training_*.txt`. The other Addition-1
TabPFN variants (v2.5 / v2.6 / v3) do not emit a comparable line, and
Addition 0 (XGBoost stack) carries no on-disk timing at all, so the
extractable wall-clock is restricted to the AutoTabPFN family.

Output: `experiment/1/wall_clock_autotabpfn_<ts>.csv` with one row per
training log (most recent log per leaf wins under the multi-log case),
columns `leaf_path, total_runtime_s, log_basename, log_mtime`.
"""
from __future__ import annotations

import csv
import datetime as _dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # experiment/
ADDITION = ROOT / "1"
PATTERN = re.compile(r"total runtime\s*=\s*([0-9]+(?:\.[0-9]+)?)s")


def _scan() -> list[dict]:
    rows: dict[Path, dict] = {}
    for log in ADDITION.rglob("_running_output/training_*.txt"):
        text = log.read_text(errors="ignore")
        m = PATTERN.search(text)
        if m is None:
            continue
        leaf = log.parent.parent
        runtime = float(m.group(1))
        mtime = log.stat().st_mtime
        prev = rows.get(leaf)
        if prev is None or mtime > prev["mtime"]:
            rows[leaf] = {
                "leaf_path": str(leaf.relative_to(ROOT)),
                "total_runtime_s": runtime,
                "log_basename": log.name,
                "log_mtime": _dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                "mtime": mtime,
            }
    return sorted((dict((k, v) for k, v in r.items() if k != "mtime")
                   for r in rows.values()), key=lambda r: r["leaf_path"])


def main() -> None:
    rows = _scan()
    if not rows:
        print("no AutoGluon `total runtime` lines found under experiment/1/")
        return
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ADDITION / f"wall_clock_autotabpfn_{ts}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["total_runtime_s"] for r in rows)
    print(f"wrote {len(rows)} rows to {out}")
    print(f"  cohort total runtime = {total:.1f}s ({total/60:.1f} min)")
    print(f"  per-leaf range = {min(r['total_runtime_s'] for r in rows):.1f}s "
          f"to {max(r['total_runtime_s'] for r in rows):.1f}s")


if __name__ == "__main__":
    main()
