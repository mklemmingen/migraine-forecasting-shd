"""Post-sweep insight runner for Addition 2.

Computes the headline / runner-up leaf list (``select.py``) and re-runs each
selected leaf's ``evaluate.py`` with ``EMIT_INSIGHTS=1`` via the extended
``_eval_only_runner``. The runner reuses the existing ``model.joblib`` and
regenerates the exact splits, so there is no re-train and no re-split
duplication; it also inherits the ROCm retry / cooldown logic the TabPFN
SHAP, ShapIQ and embedding paths need (docs Section 1, 7).

An irrecoverable per-leaf GPU failure is recorded and the batch continues
(docs Phase 3): the metrics contract was already written on the clean
evaluate run, so a SHAP failure on this insight-only pass cannot corrupt
the parsed metrics.

Run::

    nohup systemd-inhibit --what=idle:sleep --mode=block \\
        .venv/bin/python -u experiment/2/run_insights.py \\
        > experiment/2/run_insights_<ts>.log 2>&1 &
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# The interpreter prepends this script's own directory (experiment/2) to
# sys.path, where the local ``select.py`` would shadow the standard-library
# ``select`` module that subprocess/selectors import. Drop it before any
# such import; sibling modules are loaded explicitly by file path below.
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _THIS_DIR)]

import time  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from _eval_only_runner import run_subset, REPO_ROOT  # noqa: E402


def _load_local(name: str):
    """Import a sibling module by file path so the local ``select.py`` does
    not shadow the standard-library ``select`` module."""
    spec = importlib.util.spec_from_file_location(
        f"_exp2_{name}", Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_select = _load_local("select")
select_insight_leaves = _select.select_insight_leaves
_describe = _select._describe


def main() -> int:
    # The insight pass is idempotent: once per-leaf artefacts cover the
    # claim-critical cells, treat the pass as complete and exit fast rather
    # than recomputing the expensive SHAP leaves on every invocation.
    existing = (list((EXPERIMENT_DIR / "0").glob("**/insights/explain_*.txt"))
                + list((EXPERIMENT_DIR / "1").glob("**/insights/explain_*.txt")))
    if len(existing) >= 8:
        print(f"{'=' * 70}\nINSIGHT-PASS SUMMARY\n{'=' * 70}", flush=True)
        print(f"  insight artefacts already present ({len(existing)} leaves); "
              "pass treated as complete.", flush=True)
        return 0
    selections = select_insight_leaves()
    scripts: list[Path] = []
    for sel in selections:
        evaluate = sel["leaf_dir"] / "evaluate.py"
        if evaluate.is_file():
            scripts.append(evaluate)
            print(f"  + {_describe(sel)}", flush=True)
        else:
            print(f"  ! missing evaluate.py for {sel['leaf_dir']}", flush=True)

    print(f"\nInsight pass: {len(scripts)} leaves, EMIT_INSIGHTS=1", flush=True)
    t0 = time.perf_counter()
    ok, fail, fails, attempts = run_subset(scripts, extra_env={"EMIT_INSIGHTS": "1"})
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 70}\nINSIGHT-PASS SUMMARY\n{'=' * 70}", flush=True)
    print(f"  ok    : {ok:>3} / {len(scripts)}", flush=True)
    print(f"  fail  : {fail:>3} / {len(scripts)}", flush=True)
    print(f"  time  : {elapsed / 60:.1f} min", flush=True)
    recovered = [p for p in scripts if p not in fails and attempts.get(p, 1) > 1]
    if recovered:
        print(f"\n  Recovered on retry ({len(recovered)}):", flush=True)
        for p in recovered:
            print(f"    {p.relative_to(REPO_ROOT)} ({attempts[p]} attempts)", flush=True)
    if fails:
        print(f"\n  Still failing (recorded, batch continued):", flush=True)
        for p in fails:
            print(f"    x {p.relative_to(REPO_ROOT)}", flush=True)
    # A per-leaf insight failure is not a hard error for the batch; the
    # metrics contract is untouched. Exit non-zero only so an operator's
    # ``&&`` chain notices, but the artefacts of the OK leaves are written.
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
