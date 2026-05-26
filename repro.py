"""One-command paper-reproducibility scaffold.

Run from the repository root with the project venv activated:

    .venv/bin/python repro.py [--what STAGES] [--skip-evaluate] [--skip-add4]

Headline stages (default `--what` set; ~10-30 min total on a workstation):

  1. PARQUET HASHES.  Verify every engineered split parquet matches
     data/processed/_content_hashes.log. If any parquet differs the
     downstream models would silently fit on changed data; the script
     refuses to proceed and lists the drifted files.

  2. SMOKE TESTS.  Run tests/ via pytest. Fails the run if any test
     fails. This catches a regression in metrics_lib or delong before
     the expensive evaluate.py loop runs.

  3. HEADLINE EVALUATE.  Re-run the headline-cell evaluate.py scripts
     for the two paper headlines (Addition 0 migraine XGB-HP020 and
     Addition 1 headache TabPFN family at v2-6 / v3-default / v3-binary,
     all three within-family-tied variants). Skipped with --skip-evaluate.

  4. AGGREGATE.  Run experiment/2/compare.py to refresh
     experiment/2/figdata_<ts>.json from the latest results files.

  5. FIGURES.  Render docs/methodAndResults_diagramCreatorScripts/fig_*.py
     scripts that consume figdata. Failures reported, non-fatal.

Per-Addition stages (opt-in via `--what` or `--what full`):

  6. ADD2_INSIGHTS.    SHAP / ALE / ShapIQ artefacts on selected leaves
     via experiment/2/run_insights.py. ~hours on GPU (TabPFN coalition
     refits); subprocess-isolated and ROCm-retried per leaf.
  7. ADD3_TEMPORAL.    Pooled ACF + Markov + self-excitation + recurrent
     + burstiness + periodicity via experiment/3/run_temporal_analysis.py.
     Reads diary parquets; trains no models. ~seconds.
  8. ADD4_SEQUENCE.    Sequence-baseline train + evaluate for window-MLP,
     GRU, TCN across cells via experiment/4/run_all_avaliable_leaves.py.
     **~hours of GPU training**; skippable with --skip-add4 even under
     --what full.
  9. ADD5_PERSONAL.    Per-patient AUROC distribution + within-person
     C-statistic + partial-pool regimes via
     experiment/5/run_personalization.py. ~10-30 min.
 10. ADD5_EXTERNAL.    Two-fold leave-one-site-out external validation
     via experiment/5/run_external_site.py. ~30-60 min.
 11. ADD6_VALUE.       Decision curve + Brier skill + operating point
     via experiment/6/run_value.py. ~10-20 min.

Usage examples:
    .venv/bin/python repro.py                              # default headline stages
    .venv/bin/python repro.py --what figures               # only render figures
    .venv/bin/python repro.py --skip-evaluate              # skip headline evaluate
    .venv/bin/python repro.py --what hashes,tests          # only integrity + tests
    .venv/bin/python repro.py --what full                  # headline + all 6 Additions
    .venv/bin/python repro.py --what full --skip-add4      # full minus the hours-long Add-4
    .venv/bin/python repro.py --what add5_personal,add6_value  # specific Additions
    .venv/bin/python repro.py --list                       # list all stages and exit

Stage IDs (comma-separated): hashes, tests, evaluate, aggregate, figures,
add2_insights, add3_temporal, add4_sequence, add5_personal, add5_external,
add6_value.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable

# Headline cells (kept in sync with experiment/2/select.py composite_sorted).
# The headache headline is a family-level claim across three within-AUROC-tier
# TabPFN variants (v2-6 / v3-default / v3-binary tied at 0.652-0.653 within the
# composite rule's 0.02 noise tier); all three are re-run so the family claim
# is reproducible end-to-end.
HEADLINE_LEAVES = [
    REPO / "experiment/0/migraine/full_features/stacked_2xgb_meta_lr/70_30/chrono/HyperparameterTuned/single_AUROC/HP020",
    REPO / "experiment/1/headache/full_features/tabpfn/version_2-6/70_30/chrono",
    REPO / "experiment/1/headache/full_features/tabpfn/version_3-default/70_30/chrono",
    REPO / "experiment/1/headache/full_features/tabpfn/version_3-binary/70_30/chrono",
]


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def stage_hashes() -> int:
    """Verify parquet content hashes against the tracked log."""
    log = REPO / "data/processed/_content_hashes.log"
    if not log.exists():
        print(_red("  no _content_hashes.log on disk; cannot verify drift"))
        return 1
    drift = []
    for line in log.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        expected_hash, _size, rel = parts[0], parts[1], parts[2]
        path = REPO / rel
        if not path.exists():
            drift.append((rel, "MISSING"))
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if actual != expected_hash:
            drift.append((rel, f"hash {actual} != logged {expected_hash}"))
    if drift:
        print(_red(f"  {len(drift)} parquet(s) drifted from log:"))
        for rel, why in drift[:10]:
            print(f"    {rel}: {why}")
        return 1
    print(_green("  all 92 parquet hashes match the log"))
    return 0


def stage_tests() -> int:
    """Run pytest on the tests/ directory."""
    r = subprocess.run(
        [PYTHON, "-m", "pytest", "tests/", "-q"],
        cwd=REPO, capture_output=False,
    )
    return r.returncode


def stage_evaluate() -> int:
    """Run the headline evaluate.py scripts in sequence."""
    failed = []
    for leaf in HEADLINE_LEAVES:
        script = leaf / "evaluate.py"
        if not script.exists():
            failed.append((leaf, "evaluate.py missing"))
            continue
        print(f"  running {leaf.relative_to(REPO)}/evaluate.py")
        t0 = time.time()
        r = subprocess.run([PYTHON, str(script)], cwd=REPO, capture_output=False)
        if r.returncode != 0:
            failed.append((leaf, f"exit code {r.returncode}"))
        else:
            print(_green(f"  ok ({time.time()-t0:.0f}s)"))
    if failed:
        print(_red(f"  {len(failed)} headline leaf evaluate.py failed:"))
        for leaf, why in failed:
            print(f"    {leaf.relative_to(REPO)}: {why}")
        return 1
    return 0


def stage_aggregate() -> int:
    """Refresh figdata via experiment/2/compare.py."""
    print(f"  running experiment/2/compare.py")
    r = subprocess.run(
        [PYTHON, "experiment/2/compare.py"], cwd=REPO, capture_output=False,
    )
    return r.returncode


def stage_figures() -> int:
    """Render each fig_*.py that consumes figdata. Failures reported but non-fatal."""
    fig_dir = REPO / "docs/methodAndResults_diagramCreatorScripts"
    fig_files = sorted(fig_dir.glob("fig_*.py"))
    n_ok, n_fail = 0, 0
    for fp in fig_files:
        print(f"  rendering {fp.name} ...", end="", flush=True)
        t0 = time.time()
        r = subprocess.run([PYTHON, str(fp)], cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0:
            print(_green(f" ok ({time.time()-t0:.0f}s)"))
            n_ok += 1
        else:
            print(_red(" FAILED"))
            n_fail += 1
    print()
    print(f"  figures: {n_ok} ok, {n_fail} failed (out of {len(fig_files)})")
    # Non-fatal: any individual figure may fail without invalidating the run.
    return 0


def _run_driver(rel_path: str, label: str) -> int:
    """Invoke a per-Addition driver script as a subprocess.

    Streams stdout/stderr live so progress is visible during long fits.
    Returns the driver's exit code; the caller decides whether to halt.
    """
    script = REPO / rel_path
    if not script.exists():
        print(_red(f"  driver missing: {rel_path}"))
        return 1
    print(f"  {label}: running {rel_path}")
    t0 = time.time()
    r = subprocess.run([PYTHON, str(script)], cwd=REPO, capture_output=False)
    elapsed = time.time() - t0
    if r.returncode == 0:
        print(_green(f"  {label}: ok ({elapsed:.0f}s)"))
    else:
        print(_red(f"  {label}: exit code {r.returncode} ({elapsed:.0f}s)"))
    return r.returncode


def stage_add2_insights() -> int:
    """Addition 2: SHAP/ALE/ShapIQ on selected leaves. ~hours on GPU."""
    print("  estimated runtime: hours (TabPFN coalition refits on GPU)")
    return _run_driver("experiment/2/run_insights.py", "add2_insights")


def stage_add3_temporal() -> int:
    """Addition 3: pooled ACF + Markov + self-excitation + recurrent + burstiness. ~seconds."""
    return _run_driver("experiment/3/run_temporal_analysis.py", "add3_temporal")


def stage_add4_sequence() -> int:
    """Addition 4: window-MLP/GRU/TCN sequence baselines. **~hours of GPU training.**"""
    print("  estimated runtime: hours (sequence-model training on GPU)")
    print("  skippable with --skip-add4")
    return _run_driver("experiment/4/run_all_avaliable_leaves.py", "add4_sequence")


def stage_add5_personal() -> int:
    """Addition 5: per-patient AUROC distribution + within-person C-statistic + regimes. ~10-30 min."""
    return _run_driver("experiment/5/run_personalization.py", "add5_personal")


def stage_add5_external() -> int:
    """Addition 5b: two-fold leave-one-site-out external validation. ~30-60 min."""
    return _run_driver("experiment/5/run_external_site.py", "add5_external")


def stage_add6_value() -> int:
    """Addition 6: decision curve + Brier skill + operating point. ~10-20 min."""
    return _run_driver("experiment/6/run_value.py", "add6_value")


STAGES = {
    # Headline / default-on stages
    "hashes":          stage_hashes,
    "tests":           stage_tests,
    "evaluate":        stage_evaluate,
    "aggregate":       stage_aggregate,
    "figures":         stage_figures,
    # Per-Addition stages (opt-in)
    "add2_insights":   stage_add2_insights,
    "add3_temporal":   stage_add3_temporal,
    "add4_sequence":   stage_add4_sequence,
    "add5_personal":   stage_add5_personal,
    "add5_external":   stage_add5_external,
    "add6_value":      stage_add6_value,
}

HEADLINE_STAGES = ["hashes", "tests", "evaluate", "aggregate", "figures"]
ADDITION_STAGES = ["add2_insights", "add3_temporal", "add4_sequence",
                   "add5_personal", "add5_external", "add6_value"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--what", default=",".join(HEADLINE_STAGES),
                        help=f"Comma-separated stages to run. The literal "
                             f"'full' expands to headline + all Additions. "
                             f"Stages: {', '.join(STAGES.keys())}. "
                             f"Default: the headline set "
                             f"({', '.join(HEADLINE_STAGES)}).")
    parser.add_argument("--skip-evaluate", action="store_true",
                        help="Drop the headline `evaluate` stage from the run.")
    parser.add_argument("--skip-add4", action="store_true",
                        help="Drop the long-running `add4_sequence` stage "
                             "(hours of GPU training) from the run.")
    parser.add_argument("--list", action="store_true",
                        help="List every available stage with its label and exit.")
    args = parser.parse_args()

    if args.list:
        print(_bold("Headline stages (default):"))
        for s in HEADLINE_STAGES:
            print(f"  {s:18s} {STAGES[s].__doc__ or ''}")
        print()
        print(_bold("Addition stages (opt-in via --what or --what full):"))
        for s in ADDITION_STAGES:
            print(f"  {s:18s} {STAGES[s].__doc__ or ''}")
        return 0

    if args.what.strip().lower() == "full":
        chosen = HEADLINE_STAGES + ADDITION_STAGES
    else:
        chosen = [s.strip() for s in args.what.split(",") if s.strip()]

    if args.skip_evaluate:
        chosen = [s for s in chosen if s != "evaluate"]
    if args.skip_add4:
        chosen = [s for s in chosen if s != "add4_sequence"]

    bad = [s for s in chosen if s not in STAGES]
    if bad:
        print(_red(f"Unknown stage(s): {bad}. Available: {list(STAGES.keys())}"))
        return 2

    for s in chosen:
        print()
        print(_bold(f"=== stage: {s} ==="))
        rc = STAGES[s]()
        if rc != 0:
            print(_red(f"stage {s} failed (exit {rc}); halting run."))
            return rc

    print()
    print(_bold(_green(f"all requested stages ok: {chosen}")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
