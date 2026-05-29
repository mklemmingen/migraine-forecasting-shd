"""Extract per-leaf wall-clock from on-disk artefacts across Additions 0/1/4.

Two extraction channels:

1. **Precise (AutoGluon `total runtime`)**. AutoTabPFN (`version_2-5-auto`)
   training logs emit `total runtime = NNN.NNs`. Sub-second precision, but
   only present on the 4 AutoTabPFN leaves.

2. **Filename-timestamp span (all leaves)**. Every `training_*.txt` and
   `results_*.txt` embeds its creation timestamp in the filename as
   `<prefix>_YYYYMMDD_HHMMSS_<ms>_<uuid>.txt`. These are stable against
   later filesystem operations (unlike mtime, which `cp -p` and explicit
   touch can corrupt). Span = (latest results filename ts) - (earliest
   training filename ts) on the same leaf. This is an UPPER BOUND on
   wall-clock that includes any gap between separate train and eval
   invocations, so a `multi_day_gap` flag is set when the span exceeds
   12 hours (heuristic: a single train+eval cycle is shorter; a longer
   span means train and eval ran in separate sessions).

Output: `experiment/wall_clock_<ts>.csv` with one row per leaf that has
at least one results file, columns:
  leaf_path, addition, autogluon_runtime_s, span_s, earliest_training_ts,
  latest_results_ts, n_training_logs, n_results_files, multi_day_gap,
  notes.
"""
from __future__ import annotations

import csv
import datetime as _dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # experiment/
ADDITIONS = ("0", "1", "4")
AUTOGLUON_PATTERN = re.compile(r"total runtime\s*=\s*([0-9]+(?:\.[0-9]+)?)s")
FILENAME_TS_PATTERN = re.compile(r"(training|results)_(\d{8})_(\d{6})")
MULTI_DAY_THRESHOLD_S = 12 * 3600


def _parse_filename_ts(name: str) -> _dt.datetime | None:
    m = FILENAME_TS_PATTERN.search(name)
    if m is None:
        return None
    return _dt.datetime.strptime(f"{m.group(2)}_{m.group(3)}", "%Y%m%d_%H%M%S")


def _autogluon_runtime(log_path: Path) -> float | None:
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return None
    m = AUTOGLUON_PATTERN.search(text)
    return float(m.group(1)) if m else None


def _scan_addition(addition: str) -> list[dict]:
    add_root = ROOT / addition
    if not add_root.is_dir():
        return []
    rows: list[dict] = []
    seen_leaves: set[Path] = set()
    for results_dir in add_root.rglob("results"):
        if not results_dir.is_dir():
            continue
        leaf = results_dir.parent
        if leaf in seen_leaves:
            continue
        seen_leaves.add(leaf)
        results_files = sorted(results_dir.glob("results_*.txt"))
        if not results_files:
            continue
        results_ts = [_parse_filename_ts(p.name) for p in results_files]
        results_ts = [t for t in results_ts if t is not None]
        if not results_ts:
            continue
        latest_results_ts = max(results_ts)
        training_logs: list[Path] = []
        running_output = leaf / "_running_output"
        if running_output.is_dir():
            training_logs = sorted(running_output.glob("training_*.txt"))
        training_ts = [_parse_filename_ts(p.name) for p in training_logs]
        training_ts = [t for t in training_ts if t is not None]
        earliest_training_ts = min(training_ts) if training_ts else None
        autogluon_runtime_s: float | None = None
        for log in training_logs:
            rt = _autogluon_runtime(log)
            if rt is not None and (autogluon_runtime_s is None or rt > autogluon_runtime_s):
                autogluon_runtime_s = rt
        span_s: float | None = None
        multi_day_gap = False
        if earliest_training_ts is not None:
            span_s = (latest_results_ts - earliest_training_ts).total_seconds()
            multi_day_gap = span_s > MULTI_DAY_THRESHOLD_S
        notes = []
        if autogluon_runtime_s is not None:
            notes.append("autogluon-precise")
        if multi_day_gap:
            notes.append("multi_day_gap")
        if earliest_training_ts is None:
            notes.append("no_training_log")
        rows.append({
            "leaf_path": str(leaf.relative_to(ROOT)),
            "addition": addition,
            "autogluon_runtime_s": (f"{autogluon_runtime_s:.2f}"
                                    if autogluon_runtime_s is not None else ""),
            "span_s": (f"{span_s:.0f}" if span_s is not None else ""),
            "earliest_training_ts": (earliest_training_ts.isoformat()
                                     if earliest_training_ts else ""),
            "latest_results_ts": latest_results_ts.isoformat(),
            "n_training_logs": len(training_logs),
            "n_results_files": len(results_files),
            "multi_day_gap": multi_day_gap,
            "notes": ";".join(notes),
        })
    return sorted(rows, key=lambda r: r["leaf_path"])


def main() -> None:
    all_rows: list[dict] = []
    for addition in ADDITIONS:
        all_rows.extend(_scan_addition(addition))
    if not all_rows:
        print("no leaves with results files found")
        return
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / f"wall_clock_{ts}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {out}")
    print("\nbreakdown by addition:")
    for addition in ADDITIONS:
        rows = [r for r in all_rows if r["addition"] == addition]
        with_autogluon = sum(1 for r in rows if r["autogluon_runtime_s"])
        with_span = sum(1 for r in rows if r["span_s"])
        with_gap = sum(1 for r in rows if r["multi_day_gap"])
        print(f"  Addition {addition}: {len(rows)} leaves "
              f"({with_autogluon} with autogluon-precise, "
              f"{with_span} with span estimate, "
              f"{with_gap} flagged multi_day_gap)")


if __name__ == "__main__":
    main()
