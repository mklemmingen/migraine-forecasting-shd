"""Extract per-leaf training wall-clock from on-disk artefacts across Additions 0/1/4.

Three extraction channels, ordered by precision:

1. **Header/footer markers (precise, all leaves with a wrapped training log)**.
   Every `_running_output/training_*.txt` opens with
       `# Generated: YYYY-MM-DDTHH:MM:SS`
   and closes with
       `# Finished: YYYY-MM-DDTHH:MM:SS`
   wrapping the captured stdout. Delta is the precise wrapper-script
   wall-clock for the most recent training run (second resolution). The
   wrapper covers the train.py invocation including imports + fit +
   model.joblib save; it does NOT cover evaluate.py (the eval phase
   isn't wrapped by an analogous log file).

2. **AutoGluon `total runtime` (sub-second, AutoTabPFN only)**. AutoTabPFN
   (`version_2-5-auto`) emits `total runtime = NNN.NNs` inside the
   wrapped log. This is AutoGluon-internal fit time, slightly shorter
   than the header/footer span (excludes wrapper-script overhead and
   the model.joblib save).

3. **Filename-timestamp span (fallback, leaves without header markers)**.
   Embedded timestamps on `training_*.txt` and `results_*.txt` filenames
   (form `<prefix>_YYYYMMDD_HHMMSS_<ms>_<uuid>.txt`) survive later
   filesystem operations (unlike mtime, which `cp -p` and explicit
   touch can corrupt). Span = (latest results filename ts) - (earliest
   training filename ts). Upper bound, often dominated by inter-run gaps;
   flagged via `multi_day_gap` when the span exceeds 12 hours.

Addition 4 sequence leaves have no `_running_output/` directory and no
`results/` subdirectory, so none of the three channels reaches them; they
are excluded from the output.

Output: `experiment/wall_clock_<ts>.csv` with one row per leaf, columns:
  leaf_path, addition, header_duration_s (channel 1), autogluon_runtime_s
  (channel 2), span_s (channel 3), training_log_basename,
  generated_ts, finished_ts, earliest_training_ts, latest_results_ts,
  n_training_logs, n_results_files, multi_day_gap, notes (which channels
  fired).
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
GENERATED_PATTERN = re.compile(r"^# Generated:\s*(\S+)", re.MULTILINE)
FINISHED_PATTERN = re.compile(r"^# Finished:\s*(\S+)", re.MULTILINE)
ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"
MULTI_DAY_THRESHOLD_S = 12 * 3600


def _parse_filename_ts(name: str) -> _dt.datetime | None:
    m = FILENAME_TS_PATTERN.search(name)
    if m is None:
        return None
    return _dt.datetime.strptime(f"{m.group(2)}_{m.group(3)}", "%Y%m%d_%H%M%S")


def _autogluon_runtime(text: str) -> float | None:
    m = AUTOGLUON_PATTERN.search(text)
    return float(m.group(1)) if m else None


def _header_footer_duration(text: str) -> tuple[float | None, _dt.datetime | None, _dt.datetime | None]:
    """Returns (duration_s, generated_ts, finished_ts) or (None, None, None)."""
    gm = GENERATED_PATTERN.search(text)
    fm = FINISHED_PATTERN.search(text)
    if gm is None or fm is None:
        return None, None, None
    try:
        g_ts = _dt.datetime.strptime(gm.group(1), ISO_FORMAT)
        f_ts = _dt.datetime.strptime(fm.group(1), ISO_FORMAT)
    except ValueError:
        return None, None, None
    return (f_ts - g_ts).total_seconds(), g_ts, f_ts


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
        # Most-recent training log = the run that produced the current model
        most_recent_log: Path | None = None
        if training_logs:
            most_recent_log = max(training_logs, key=lambda p: _parse_filename_ts(p.name) or _dt.datetime.min)
        header_duration_s: float | None = None
        generated_ts: _dt.datetime | None = None
        finished_ts: _dt.datetime | None = None
        autogluon_runtime_s: float | None = None
        if most_recent_log is not None:
            try:
                text = most_recent_log.read_text(errors="ignore")
            except OSError:
                text = ""
            header_duration_s, generated_ts, finished_ts = _header_footer_duration(text)
            autogluon_runtime_s = _autogluon_runtime(text)
        span_s: float | None = None
        multi_day_gap = False
        if earliest_training_ts is not None:
            span_s = (latest_results_ts - earliest_training_ts).total_seconds()
            multi_day_gap = span_s > MULTI_DAY_THRESHOLD_S
        notes = []
        if header_duration_s is not None:
            notes.append("header-precise")
        if autogluon_runtime_s is not None:
            notes.append("autogluon-precise")
        if multi_day_gap:
            notes.append("multi_day_gap")
        if earliest_training_ts is None:
            notes.append("no_training_log")
        rows.append({
            "leaf_path": str(leaf.relative_to(ROOT)),
            "addition": addition,
            "header_duration_s": (f"{header_duration_s:.0f}"
                                  if header_duration_s is not None else ""),
            "autogluon_runtime_s": (f"{autogluon_runtime_s:.2f}"
                                    if autogluon_runtime_s is not None else ""),
            "span_s": (f"{span_s:.0f}" if span_s is not None else ""),
            "training_log_basename": most_recent_log.name if most_recent_log else "",
            "generated_ts": generated_ts.isoformat() if generated_ts else "",
            "finished_ts": finished_ts.isoformat() if finished_ts else "",
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
        with_header = sum(1 for r in rows if r["header_duration_s"])
        with_autogluon = sum(1 for r in rows if r["autogluon_runtime_s"])
        with_gap = sum(1 for r in rows if r["multi_day_gap"])
        header_vals = [float(r["header_duration_s"]) for r in rows if r["header_duration_s"]]
        if header_vals:
            header_vals.sort()
            stats = (f"median {header_vals[len(header_vals)//2]:.0f}s, "
                     f"range {header_vals[0]:.0f}s-{header_vals[-1]:.0f}s, "
                     f"total {sum(header_vals)/3600:.1f}h")
        else:
            stats = "no header-precise rows"
        print(f"  Addition {addition}: {len(rows)} leaves "
              f"({with_header} header-precise, {with_autogluon} autogluon-precise, "
              f"{with_gap} multi_day_gap)")
        print(f"    header-precise training duration: {stats}")


if __name__ == "__main__":
    main()
