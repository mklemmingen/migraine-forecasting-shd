"""Output rotation for the aggregator.

Each run produces ~7 files (the current comparison HTML plus its
supporting PNGs and the legacy interactive HTML). Left in place they
quickly clutter ``experiment/``; this module moves prior outputs into
dated ``experiment/results/YYYY-MM-DD/`` subfolders so the active
working directory stays one click away from the current artefacts.
"""
from datetime import datetime
from pathlib import Path

OUTPUT_PATTERNS = (
    "results_*.html",
    "comparison_*.html",
    "comparison_*.csv",
    # PNG (HTML preview) + vector PDF (journal asset) for every figure.
    "venn_counts_*.png", "venn_counts_*.pdf",
    "venn_names_*.png",  "venn_names_*.pdf",
    "tree_*.png",        "tree_*.pdf",
    "cd_*.png",          "cd_*.pdf",
    "perfprofile_*.png", "perfprofile_*.pdf",
    "slopegraph_*.png",  "slopegraph_*.pdf",
)


def _archive_dest_for(src_path, dest_root):
    """Return ``<dest_root>/<YYYY-MM-DD>/<basename>`` for an aggregator
    output file, where the date is taken from the file's own mtime.
    """
    day_str = datetime.fromtimestamp(src_path.stat().st_mtime).strftime("%Y-%m-%d")
    day_dir = dest_root / day_str
    day_dir.mkdir(exist_ok=True)
    return day_dir / src_path.name


def _migrate_flat_archive(archive_dir):
    """One-time bring-up: if any aggregator outputs sit directly under
    ``experiment/results/`` (from a flat-archive layout that predates
    the dated subfolders), file them into their YYYY-MM-DD subfolder.
    """
    migrated = 0
    for pattern in OUTPUT_PATTERNS:
        for src in archive_dir.glob(pattern):
            if not src.is_file():
                continue
            dest = _archive_dest_for(src, archive_dir)
            if dest == src:
                continue
            if dest.exists():
                dest.unlink()
            src.rename(dest)
            migrated += 1
    if migrated:
        print(f"Migrated {migrated} flat-archived file{'s' if migrated != 1 else ''} "
              f"into {archive_dir}/YYYY-MM-DD/ subfolders")


def archive_previous_outputs(latest_dir: Path, archive_dir: Path):
    """Move any prior aggregator outputs from ``latest_dir`` into
    ``archive_dir`` under dated subfolders.

    Each archived file is filed under ``archive_dir/YYYY-MM-DD/``
    (ISO-8601 calendar date) inferred from its mtime. This keeps the
    archive browsable by day; dropping seven outputs at a time into a
    single flat directory rapidly becomes unreadable, but per-day
    folders preserve the run-grouping a reader actually cares about.

    The scan over ``latest_dir`` is non-recursive so per-leaf ``.txt``
    files inside ``experiment/<.../>results/`` are untouched.
    """
    archive_dir.mkdir(exist_ok=True)
    _migrate_flat_archive(archive_dir)
    moved = 0
    per_day_counts: dict[str, int] = {}
    for pattern in OUTPUT_PATTERNS:
        for src in latest_dir.glob(pattern):
            if not src.is_file():
                continue
            dest = _archive_dest_for(src, archive_dir)
            if dest.exists():
                # Re-run with identical timestamp; keep the new version.
                dest.unlink()
            day_str = dest.parent.name
            src.rename(dest)
            moved += 1
            per_day_counts[day_str] = per_day_counts.get(day_str, 0) + 1
    if moved:
        per_day_summary = ", ".join(
            f"{day}={n}" for day, n in sorted(per_day_counts.items())
        )
        print(f"Archived {moved} prior output file{'s' if moved != 1 else ''} "
              f"into {archive_dir}/YYYY-MM-DD/ ({per_day_summary})")
