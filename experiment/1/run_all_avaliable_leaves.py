"""
run_all_avaliable_leaves.py - Sequential runner for every leaf under this addition.

Phase 1 - every train.py, split per target (headache then migraine).
Phase 2 - every evaluate.py + evaluate_cv.py, split per target, after all training.

Phase 2 runs only after Phase 1 finishes (across all targets), so every
model.joblib needed by an evaluator already exists. Within each phase, leaves
are grouped by target so it's clear which cohort is being processed.

Failures don't abort: each leaf's exit code is recorded and the runner moves
on, with a per-target / per-script pass/fail summary at the end.

stdout/stderr from each script streams live to the parent terminal so progress
(and warnings) are visible during long stacking fits - Ctrl-C interrupts the
current leaf and the runner exits.

Run: `.venv/bin/python experiment/0/run_all_avaliable_leaves.py`
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ADDITION_ROOT = Path(__file__).resolve().parent      # experiment/<addition>/
PYTHON = sys.executable                              # use the launching interpreter

# Target sub-trees, in run order. A target is missing if its directory is empty.
TARGETS = ("headache", "migraine")


def find_in(target: str, name: str) -> list[Path]:
    """All `name`-named scripts under experiment/<addition>/<target>/, sorted."""
    target_root = ADDITION_ROOT / target
    if not target_root.is_dir():
        return []
    return sorted(
        p for p in target_root.rglob(name)
        if "__pycache__" not in p.parts
    )


def run(script: Path, idx: int, total: int) -> tuple[bool, float]:
    """Run one script as a subprocess; stream its output. Return (ok, seconds)."""
    rel = script.relative_to(ADDITION_ROOT)
    print(f"\n{'=' * 70}\n[{idx}/{total}] {rel}\n{'=' * 70}")
    t0 = time.perf_counter()
    result = subprocess.run([PYTHON, str(script)])
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    print(f"  → {'ok' if ok else f'FAIL (exit {result.returncode})'}  ({elapsed:.1f}s)")
    return ok, elapsed


def sub_phase(label: str, scripts: list[Path]) -> dict:
    """Run a list of scripts, print a sub-phase header, collect stats."""
    print(f"\n\n--- {label}  -  {len(scripts)} script{'s' if len(scripts) != 1 else ''} ---")
    stats = {"ok": 0, "fail": 0, "secs": 0.0, "fails": []}
    for i, script in enumerate(scripts, 1):
        ok, secs = run(script, i, len(scripts))
        stats["secs"] += secs
        if ok:
            stats["ok"] += 1
        else:
            stats["fail"] += 1
            stats["fails"].append(script.relative_to(ADDITION_ROOT))
    return stats


def phase_split_by_target(label: str, names: tuple[str, ...]) -> dict:
    """Run a phase as one sub-phase per target, accumulating combined stats."""
    print(f"\n\n############################################################")
    print(f"### {label}")
    print(f"############################################################")
    combined = {"ok": 0, "fail": 0, "secs": 0.0, "fails": []}
    for target in TARGETS:
        scripts: list[Path] = []
        for name in names:
            scripts.extend(find_in(target, name))
        if not scripts:
            print(f"\n  (no {' / '.join(names)} found under {target}/, skipping)")
            continue
        stats = sub_phase(f"target = {target}", scripts)
        combined["ok"] += stats["ok"]
        combined["fail"] += stats["fail"]
        combined["secs"] += stats["secs"]
        combined["fails"].extend(stats["fails"])
    return combined


def main() -> int:
    print(f"Addition root: {ADDITION_ROOT}")
    print(f"Interpreter:   {PYTHON}")
    print(f"Targets:       {', '.join(TARGETS)}")

    train_stats = phase_split_by_target("Phase 1 - train.py", ("train.py",))
    eval_stats = phase_split_by_target(
        "Phase 2 - evaluate.py + evaluate_cv.py",
        ("evaluate.py", "evaluate_cv.py"),
    )

    print(f"\n\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for label, s in (("Phase 1 train  ", train_stats), ("Phase 2 evaluate", eval_stats)):
        print(f"  {label}: {s['ok']:>3} ok / {s['fail']:>2} fail  ({s['secs']:.1f}s total)")
    fails = train_stats["fails"] + eval_stats["fails"]
    if fails:
        print(f"\n  Failed scripts:")
        for f in fails:
            print(f"    ✗ {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
