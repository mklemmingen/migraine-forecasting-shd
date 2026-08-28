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
# sys.path. The module that used to shadow the standard-library ``select``
# from here has been renamed to ``leaf_selection``, so that hazard is gone;
# the directory is still dropped as defence in depth, and sibling modules
# are loaded explicitly by file path below.
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _THIS_DIR)]

import argparse  # noqa: E402
import time  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from _eval_only_runner import run_subset, REPO_ROOT  # noqa: E402


def _load_local(name: str):
    """Import a sibling module by file path, independent of sys.path."""
    spec = importlib.util.spec_from_file_location(
        f"_exp2_{name}", Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_select = _load_local("leaf_selection")
select_insight_leaves = _select.select_insight_leaves
_describe = _select._describe


def _has_insights(leaf_dir: Path) -> bool:
    ins = leaf_dir / "insights"
    return ins.is_dir() and any(ins.glob("explain_*.txt"))


def main() -> int:
    # Per-leaf idempotent by default: leaves that already carry insight
    # artefacts (insights/explain_*.txt) are skipped, so a re-run fills coverage
    # gaps without recomputing the expensive SHAP leaves already on disk.
    # ``--force`` overrides the skip (e.g., to re-run after emit_insights has
    # been extended with new artefacts like the SHAP-matrix .npz stash);
    # optional positional leaf paths narrow the run to just those leaves, which
    # must still be members of the report's selected-leaves set.
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("leaves", nargs="*", type=Path, metavar="LEAF",
                        help="Specific leaf directories to (re-)run. Default: all "
                             "leaves picked by select_insight_leaves(). Any path "
                             "not in that selection is rejected.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run leaves that already carry insight artefacts. "
                             "Without this, leaves whose insights/ already has an "
                             "explain_*.txt are skipped (the default idempotency).")
    args = parser.parse_args()

    full_selections = select_insight_leaves()
    if args.leaves:
        by_leaf = {Path(s["leaf_dir"]).resolve(): s for s in full_selections}
        selections = []
        for p in args.leaves:
            sel = by_leaf.get(Path(p).resolve())
            if sel is None:
                print(f"  ! {p} is not in the selected-leaves set; ignoring.",
                      flush=True)
                continue
            selections.append(sel)
        if not selections:
            print("No matching leaves; nothing to do.", flush=True)
            return 0
    else:
        selections = full_selections

    scripts: list[Path] = []
    seen: set = set()
    skipped = 0
    for sel in selections:
        leaf = sel["leaf_dir"]
        if leaf in seen:
            continue
        seen.add(leaf)
        if not args.force and _has_insights(leaf):
            skipped += 1
            continue
        evaluate = leaf / "evaluate.py"
        if evaluate.is_file():
            scripts.append(evaluate)
            print(f"  + {_describe(sel)}", flush=True)
        else:
            print(f"  ! missing evaluate.py for {leaf}", flush=True)

    force_note = " (--force: re-running insighted leaves)" if args.force else ""
    print(f"\nInsight pass: {len(scripts)} leaves to run, {skipped} already "
          f"insighted (skipped){force_note}, EMIT_INSIGHTS=1", flush=True)
    if not scripts:
        print("nothing to do.", flush=True)
        return 0
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
