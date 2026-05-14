"""
_eval_only_runner.py - run every leaf's evaluate.py + evaluate_cv.py
across every addition under experiment/, without re-running train.py.

The runner re-executes only the evaluate phase against the existing
model.joblib for each leaf, regenerating the per-leaf result.txt
files. This is the path to take when a change to the evaluator
template or the metric set needs to propagate across the sweep
without paying the train-phase GPU cost.

Layout assumption: ``experiment/<N>/<target>/<feature_set>/<arch>/.../
{evaluate,evaluate_cv}.py``.  Anything under a `_`-prefixed folder is
skipped (test scaffolding, smoke runners, _run_logs, etc.).

Failures don't abort: each leaf's exit code is recorded and the runner
moves on. Stdout/stderr streams live to the parent terminal so warnings
and the per-leaf threshold readout are visible in real time.

Run::

    nohup systemd-inhibit --what=idle:sleep --mode=block \\
        .venv/bin/python -u experiment/_eval_only_runner.py \\
        > addition_eval_only_<ts>.txt 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT       = EXPERIMENT_ROOT.parent
PYTHON          = sys.executable

# Additions are top-level numeric folders under experiment/. New additions
# (experiment/2, 3, ...) get picked up automatically.
ADDITIONS = sorted(
    p.name for p in EXPERIMENT_ROOT.iterdir()
    if p.is_dir() and p.name.isdigit()
)

# ROCm stability + retry settings (mirroring _run_logs/_rerun_failed.py).
# Native crashes inside PyTorch+ROCm during TabPFN inference/refit cannot
# be caught from Python; the only remediation is subprocess respawn.
_MAX_ATTEMPTS     = 3
_COOLDOWN_SECONDS = 8
_ROCM_ENV = {
    "PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:128",
    "MIOPEN_DEBUG_DISABLE_FIND_DB": "1",
}


def find_leaves_in(addition: str, name: str) -> list[Path]:
    """All ``name``-named scripts under experiment/<addition>/, sorted,
    skipping any path that has a `_`-prefixed component."""
    addition_root = EXPERIMENT_ROOT / addition
    if not addition_root.is_dir():
        return []
    out = []
    for p in addition_root.rglob(name):
        if any(part.startswith("_") for part in p.parts):
            continue
        if "__pycache__" in p.parts:
            continue
        out.append(p)
    return sorted(out)


def _run_once(script: Path, attempt: int) -> tuple[bool, int, float]:
    """One subprocess attempt of a leaf script. Returns (ok, exit_code, elapsed)."""
    env = dict(os.environ)
    env.update(_ROCM_ENV)
    t0 = time.perf_counter()
    print(f"\n----- attempt {attempt}/{_MAX_ATTEMPTS} ----- "
          f"env: PYTORCH_HIP_ALLOC_CONF={env['PYTORCH_HIP_ALLOC_CONF']} "
          f"MIOPEN_DEBUG_DISABLE_FIND_DB={env['MIOPEN_DEBUG_DISABLE_FIND_DB']}",
          flush=True)
    result = subprocess.run([PYTHON, "-u", str(script)], env=env)
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    tag = "ok" if ok else f"FAIL (exit {result.returncode})"
    print(f"  -> {tag}  ({elapsed:.1f}s)", flush=True)
    return ok, result.returncode, elapsed


def run(script: Path, idx: int, total: int) -> tuple[bool, int]:
    """Retry-aware leaf runner. Up to ``_MAX_ATTEMPTS`` subprocess attempts
    with a cool-down + cache-flushing ROCm env between attempts."""
    rel = script.relative_to(REPO_ROOT)
    print(f"\n{'=' * 70}\n[{idx}/{total}] {rel}\n{'=' * 70}", flush=True)
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        ok, _, _ = _run_once(script, attempt)
        if ok:
            return True, attempt
        if attempt < _MAX_ATTEMPTS:
            print(f"  cool-down {_COOLDOWN_SECONDS}s before retry...", flush=True)
            time.sleep(_COOLDOWN_SECONDS)
    return False, _MAX_ATTEMPTS


def main() -> int:
    print(f"Repo root:    {REPO_ROOT}", flush=True)
    print(f"Interpreter:  {PYTHON}", flush=True)
    print(f"Additions:    {', '.join(ADDITIONS) if ADDITIONS else '(none)'}", flush=True)
    print(f"Mode:         evaluate-only (skips train.py - "
          f"reuses existing model.joblib files)", flush=True)

    # Collect everything first so the summary is accurate before we burn
    # GPU time. evaluate_cv.py is grouped second to keep per-addition
    # ordering: all hold-out evaluators then all CV evaluators within an
    # addition.
    plan: list[Path] = []
    for addition in ADDITIONS:
        plan.extend(find_leaves_in(addition, "evaluate.py"))
        plan.extend(find_leaves_in(addition, "evaluate_cv.py"))

    n_eval    = sum(1 for p in plan if p.name == "evaluate.py")
    n_eval_cv = sum(1 for p in plan if p.name == "evaluate_cv.py")
    print(f"Plan:         {len(plan)} scripts "
          f"({n_eval} evaluate.py + {n_eval_cv} evaluate_cv.py)", flush=True)

    print(f"Retry policy: up to {_MAX_ATTEMPTS} attempts per leaf, "
          f"{_COOLDOWN_SECONDS}s cool-down between attempts (mirrors "
          f"_rerun_failed.py).", flush=True)

    ok_count = 0
    fail_count = 0
    fails: list[Path] = []
    attempts_used: dict[Path, int] = {}
    t_total_start = time.perf_counter()

    for i, script in enumerate(plan, 1):
        ok, attempts = run(script, i, len(plan))
        attempts_used[script] = attempts
        if ok:
            ok_count += 1
        else:
            fail_count += 1
            fails.append(script)

    total_elapsed = time.perf_counter() - t_total_start
    print(f"\n\n{'=' * 70}\nEVAL-ONLY SUMMARY\n{'=' * 70}", flush=True)
    print(f"  ok    : {ok_count:>4} / {len(plan)}", flush=True)
    print(f"  fail  : {fail_count:>4} / {len(plan)}", flush=True)
    print(f"  time  : {total_elapsed/60:.1f} min", flush=True)

    retried_ok = [p for p in plan if p not in fails and attempts_used.get(p, 1) > 1]
    if retried_ok:
        print(f"\n  Recovered on retry ({len(retried_ok)}):", flush=True)
        for p in retried_ok:
            print(f"    {p.relative_to(REPO_ROOT)}  "
                  f"(took {attempts_used[p]} attempts)", flush=True)
    if fails:
        print(f"\n  Still failing after {_MAX_ATTEMPTS} attempts:", flush=True)
        for f in fails:
            print(f"    x {f.relative_to(REPO_ROOT)}", flush=True)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
