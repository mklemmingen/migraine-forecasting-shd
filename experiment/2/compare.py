"""Cross-leaf comparison artefacts for Addition 2.

Two outputs the paper's discussion cites that belong to no single leaf
(docs Sections 1, 5, 8, claims 2-3):

1. ``comparison_shap_<ts>.html`` - the headline-vs-runner-up SHAP-ranking
   diff across the seven cells. It tests whether a near-tied runner-up of a
   *different architecture family* wins on the same features as the headline
   (research question 1). At least one Addition-0 XGBoost vs Addition-1
   TabPFN pair is required (the benchmark's core question, docs Section 5/6);
   the script asserts the property holds across the selected cells.

2. ``park_or_check_<ts>.html`` - the mean |SHAP| ranking on the
   migraine/park cells against Park et al. 2016 Table-4 odds ratios
   (stress, hormonal_changes, noise, alcohol, overeating, travel). It is the
   clean (EPV 33.5) external check on whether the model recovered the
   established trigger structure (claim 2).

Both read each selected leaf's latest ``insights/explain_<ts>.txt`` (the
machine-parseable summary the insight pass writes), so this script depends
only on artefacts already on disk, not on re-running any model.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

# Drop this script's own directory (experiment/2) from sys.path before
# importing scipy: the local ``select.py`` would otherwise shadow the
# standard-library ``select`` module that scipy's subprocess import needs.
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _THIS_DIR)]

from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from _eval._html_to_pdf import html_to_pdf  # noqa: E402

_PREV_CACHE: dict = {}


def _test_prevalence(target: str, datasplit: str, splittype: str):
    """Positive-class prevalence of a leaf's hold-out test set, the no-skill
    AUPRC baseline for that exact (target, ratio, split). Cached; the label
    column is ``migraine_target`` in both pipelines (next-day headache in the
    headache dataset, next-day migraine in the migraine dataset). Returns None
    when the test parquet is absent."""
    key = (target, datasplit, splittype)
    if key in _PREV_CACHE:
        return _PREV_CACHE[key]
    p = (REPO_ROOT / "data" / "processed" / target / datasplit / splittype
         / "diary_test.parquet")
    prev = None
    if p.is_file():
        try:
            prev = float(pd.read_parquet(p, columns=["migraine_target"])
                         ["migraine_target"].mean())
        except Exception:
            prev = None
    _PREV_CACHE[key] = prev
    return prev

_FIG_DIR = Path(__file__).resolve().parent / "figures"


def _load_local(name: str):
    """Import a sibling module by file path without putting this directory
    on ``sys.path``. The local ``select.py`` would otherwise shadow the
    standard-library ``select`` module (which scipy/subprocess import),
    breaking any later third-party import."""
    spec = importlib.util.spec_from_file_location(
        f"_exp2_{name}", Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_select = _load_local("select")
select_insight_leaves = _select.select_insight_leaves

# The five comparison plotters live in the sibling _figures module so both this
# driver and the docs per-figure scripts render from one source. Loaded by file
# path (not via sys.path) for the same reason select.py is - to keep this
# directory off sys.path and avoid shadowing stdlib modules scipy imports.
_figures = _load_local("_figures")
split_auroc_figure = _figures.split_auroc_figure
auprc_lift_figure = _figures.auprc_lift_figure
calib_slope_figure = _figures.calib_slope_figure
cross_arch_figure = _figures.cross_arch_figure
park_scatter_figure = _figures.park_scatter_figure

from _eval._archival import archive_previous_outputs  # noqa: E402

# Compare outputs rotate into experiment/2/results/YYYY-MM-DD/ with the same
# dated-archive algorithm the aggregator uses; the frozen figure data rotates
# alongside the reports.
COMPARE_PATTERNS = ("comparison_shap_*.html", "comparison_shap_*.pdf",
                    "park_or_check_*.html", "park_or_check_*.pdf",
                    "figdata_*.json")

# Park et al. 2016 Table 4 [park2016shd, Tab. 4, p. 8] stepwise-selected
# trigger odds ratios; the park feature loader maps the two Korean hormonal
# sub-fields into hormonal_changes_today.
PARK_TABLE4_OR = {
    "stress_today": 1.8,
    "hormonal_changes_today": 3.5,
    "noise_today": 2.8,
    "alcohol_today": 2.5,
    "overeating_today": 2.4,
    "travel_today": 6.4,
}


def _read_ranking_csv(path: Path) -> list[tuple[str, float]]:
    """Read the full ``ranking_<ts>.csv`` matching an explain_<ts>.txt path.

    Returns the complete ranked feature list; the text summary keeps only the
    top-10, so the cross-leaf, Park-OR and prodromal checks read the CSV to
    see every feature's attribution. Returns an empty list when absent."""
    ts = path.stem.replace("explain_", "")
    csv_path = path.parent / f"ranking_{ts}.csv"
    if not csv_path.is_file():
        return []
    out: list[tuple[str, float]] = []
    for line in csv_path.read_text().splitlines():
        if line.startswith("#") or line.startswith("rank,"):
            continue
        parts = line.split(",")
        if len(parts) == 3:
            try:
                out.append((parts[1], float(parts[2])))
            except ValueError:
                pass
    return out


def parse_explain(path: Path) -> dict | None:
    """Parse a leaf's latest explain_<ts>.txt into ranking + metadata.

    Returns ``{ranking: [(feature, value), ...], metric, arch_family,
    attribution}`` or ``None`` when the file is absent/empty. The ranking is
    taken from the full ``ranking_<ts>.csv`` when present (the text summary
    truncates to the top-10), otherwise from the txt's "Top features" block.
    """
    if path is None or not path.is_file():
        return None
    lines = path.read_text().splitlines()
    ranking: list[tuple[str, float]] = []
    metric = "mean |SHAP|"
    arch_family = attribution = "?"
    in_block = False
    for line in lines:
        s = line.strip()
        if s.startswith("architecture family"):
            arch_family = s.split(":", 1)[1].strip()
        elif s.startswith("attribution"):
            attribution = s.split(":", 1)[1].strip()
        elif s.startswith("Top features by"):
            metric = s.split("by", 1)[1].rstrip(":").strip()
            in_block = True
            continue
        elif in_block:
            if s.startswith("-" * 5) or not s:
                in_block = False
                continue
            # "  1. feature_name            0.0123"
            body = s.split(".", 1)[1].strip() if "." in s.split()[0] else s
            parts = body.rsplit(None, 1)
            if len(parts) == 2:
                try:
                    ranking.append((parts[0].strip(), float(parts[1])))
                except ValueError:
                    pass
    full_ranking = _read_ranking_csv(path)
    if full_ranking:
        ranking = full_ranking
    return {"ranking": ranking, "metric": metric,
            "arch_family": arch_family, "attribution": attribution}


def latest_explain(leaf_dir: Path) -> Path | None:
    files = sorted((leaf_dir / "insights").glob("explain_*.txt")) if (leaf_dir / "insights").is_dir() else []
    return files[-1] if files else None


def resolve_insighted(sel: dict, all_rows: list[dict]) -> dict:
    """Map a selection to a leaf that actually has insight artefacts.

    ``select.py`` picks the composite-best leaf per cell, but the insight
    pass may have run a different leaf of the same cell (the metrics shift the
    headline between runs). When the selected leaf has no ``explain_*.txt``,
    fall back to the composite-best leaf of the same ``(target, feature_set,
    split_type, family)`` that does, so the comparison reads the evidence on
    disk rather than reporting a spurious miss. The split type is part of the
    key: chronological and stratified are different scientific regimes, so a
    chrono selection must never borrow a stratified leaf's attributions (that
    would file one split's evidence under another and collide with the real
    occupant of the target bucket). Degenerate-calibration leaves are excluded
    from the fallback: SHAP on an inverted or wildly mis-scaled calibrated
    probability explains noise, so a substitute must clear the same
    calibration guard the selection applies - never present such a leaf's
    attributions as a stand-in. The fallback uses ``composite_sorted`` (not
    raw AUROC) so it ranks candidates exactly as the selection does. Returns
    the original sel when no trustworthy insighted leaf of that
    cell-split-family exists (the caller then reports the miss / pending).
    """
    if latest_explain(sel["leaf_dir"]) is not None:
        return sel
    same = [r for r in all_rows
            if r["target"] == sel["target"]
            and r["feature_set"] == sel["feature_set"]
            and r["splittype"] == sel["splittype"]
            and r["family"] == sel["family"]
            and latest_explain(r["leaf_dir"]) is not None
            and not _select._calibration_degenerate(r)]
    if not same:
        return sel
    best = _select.composite_sorted(same)[0]
    return dict(best, role=sel["role"])


def _rank_overlap_table(headline: dict, runner: dict, top_n: int = 10) -> str:
    """HTML table aligning the two leaves' top-N feature rankings + the
    Spearman rank correlation over their shared features."""
    h_rank = {f: i for i, (f, _) in enumerate(headline["ranking"])}
    r_rank = {f: i for i, (f, _) in enumerate(runner["ranking"])}
    shared = [f for f in h_rank if f in r_rank]
    rho = None
    if len(shared) >= 3:
        rho, _ = spearmanr([h_rank[f] for f in shared], [r_rank[f] for f in shared])
    rows = []
    for i in range(top_n):
        hf = headline["ranking"][i][0] if i < len(headline["ranking"]) else ""
        rf = runner["ranking"][i][0] if i < len(runner["ranking"]) else ""
        rows.append(f"<tr><td>{i + 1}</td><td>{hf}</td><td>{rf}</td></tr>")
    rho_str = f"{rho:+.3f}" if rho is not None else "n/a"
    if rho is None:
        verdict = ""
    else:
        mag = abs(rho)
        strength = ("strong" if mag >= 0.7 else "moderate" if mag >= 0.4
                    else "weak")
        direction = "agreement" if rho >= 0 else "disagreement"
        verdict = (f" - {strength} cross-architecture {direction} on feature "
                   "ordering")
    return (f"<p>Spearman rank correlation over all {len(shared)} shared "
            f"features: <b>{rho_str}</b>{verdict}.</p>"
            f"<table><tr><th>rank</th><th>headline ({headline['arch_family']})</th>"
            f"<th>runner-up ({runner['arch_family']})</th></tr>"
            + "".join(rows) + "</table>")


SPLIT_LABELS = {
    "chrono": "Chronological (forecasting-honest)",
    "stratified": "Stratified (leakage contrast - not deployable by construction)",
    "patient": "Patient hold-out (generalisation to unseen patients)",
}


def _leaf_meta(sel: dict) -> str:
    """Full leaf provenance: architecture + version, hyperparameter-tuning
    configuration, data-split ratio + split type, and the hold-out metrics
    the multi-metric selection used (AUROC, AUPRC, calibration slope)."""
    auprc = f"{sel['auprc_mean']:.3f}" if sel.get("auprc_mean") is not None else "n/a"
    calib = f"{sel['calib_slope']:+.2f}" if sel.get("calib_slope") is not None else "n/a"
    ver = f"/{sel['version']}" if sel.get("version") else ""
    if sel.get("hp_strategy"):
        hp = sel["hp_strategy"] + (f"/{sel['hp_variant']}" if sel.get("hp_variant") else "")
    else:
        hp = "NonHP (library defaults)"
    return (f"<b>{sel['architecture']}{ver}</b> [{hp}] | split "
            f"{sel['datasplit']}/{sel['splittype']} | hold-out AUROC "
            f"{sel['auroc_mean']:.3f}, AUPRC {auprc}, calib {calib}")


def _latest_img(leaf_dir: Path, prefix: str) -> Path | None:
    """Latest ``insights/<prefix>_<ts>.png`` for a leaf, or None."""
    ins = leaf_dir / "insights"
    files = sorted(ins.glob(f"{prefix}_*.png")) if ins.is_dir() else []
    return files[-1] if files else None


def _embed_png(path: Path | None, max_width: int = 460) -> str:
    """Inline a PNG as a base64 data-URI so the report stays self-contained
    when archived into a dated subfolder (relative paths would break)."""
    if path is None or not path.is_file():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<img src="data:image/png;base64,{b64}" loading="lazy" '
            f'style="max-width:{max_width}px;width:100%;border:1px solid #ddd;'
            f'border-radius:4px;margin:4px 6px 4px 0">')


def _leaf_figures(sel: dict, label: str) -> str:
    """Embed a leaf's SHAP beeswarm (distributional summary) and bar chart."""
    bee = _embed_png(_latest_img(sel["leaf_dir"], "shap_beeswarm"))
    bar = _embed_png(_latest_img(sel["leaf_dir"], "shap_bar"))
    if not bee and not bar:
        return ""
    cap = (f"{label}: {sel['family']} {sel['architecture']} "
           f"({sel['datasplit']}/{sel['splittype']})")
    return (f"<div style='display:inline-block;vertical-align:top;max-width:480px'>"
            f"<div style='font-size:0.82rem;color:#555;margin-top:6px'>{cap}</div>"
            f"{bee}{bar}</div>")


def _is_cross_arch(h, r, h_sel, r_sel) -> bool:
    """True when a (headline, runner-up) pair is a genuine cross-architecture
    contrast: both insighted, families differ (xgboost vs tabpfn/autotabpfn),
    and they are different leaves (a same-leaf pair would draw a model against
    itself). Shared by the report cell and the frozen figure data."""
    if h is None or r is None:
        return False
    fams = {h["arch_family"], r["arch_family"]}
    return ("xgboost" in fams and bool({"tabpfn", "autotabpfn"} & fams)
            and h_sel["leaf_dir"] != r_sel["leaf_dir"])


def _park_ranks(ex):
    """(shared, or_rank, shap_rank, rho, p) for the Park check from a parsed
    explain dict; rho and p are None when fewer than three Park triggers are
    shared (the scatter is then suppressed, but the report still tables the
    triggers). The p-value matters: with only ~6 shared triggers any rho is
    highly uncertain, so it must be reported alongside the coefficient.
    Shared by the report's Park block and the frozen figure data."""
    shap_rank = {f: i + 1 for i, (f, _) in enumerate(ex["ranking"])}
    or_rank = {f: i + 1 for i, f in enumerate(
        sorted(PARK_TABLE4_OR, key=PARK_TABLE4_OR.get, reverse=True))}
    shared = [f for f in PARK_TABLE4_OR if f in shap_rank]
    rho = p = None
    if len(shared) >= 3:
        rho, p = spearmanr([shap_rank[f] for f in shared],
                           [or_rank[f] for f in shared])
        rho, p = float(rho), float(p)
    return shared, or_rank, shap_rank, rho, p


# Fields each headline carries into the frozen figure data; prevalence is baked
# in for headline rows so the AUPRC-lift plotter reads no parquet.
_HEADLINE_FIELDS = ("role", "target", "feature_set", "splittype", "datasplit",
                    "family", "architecture", "auroc_mean", "auroc_lo", "auroc_hi",
                    "auprc_mean", "auprc_lo", "auprc_hi", "calib_slope")


def _gather_figdata(raw_selections, selections) -> dict:
    """Build the figure-ready findings dict that both the report figures and the
    docs per-figure scripts render from. Parses each cell's SHAP rankings once
    and resolves the Park trigger ranks, so the frozen JSON is self-contained.
    """
    headlines = []
    for s in raw_selections:
        entry = {k: s.get(k) for k in _HEADLINE_FIELDS}
        if entry.get("role") == "headline":
            entry["prevalence"] = _test_prevalence(
                s["target"], s["datasplit"], s["splittype"])
        headlines.append(entry)

    by_split: dict = {}
    for sel in selections:
        by_split.setdefault(sel["splittype"], {}).setdefault(
            (sel["target"], sel["feature_set"]), {})[sel["role"]] = sel
    cross_arch = []
    for cells in by_split.values():
        for (target, fset), roles in cells.items():
            h_sel, r_sel = roles.get("headline"), roles.get("runner_up")
            if not h_sel or not r_sel:
                continue
            h = parse_explain(latest_explain(h_sel["leaf_dir"]))
            r = parse_explain(latest_explain(r_sel["leaf_dir"]))
            if not _is_cross_arch(h, r, h_sel, r_sel):
                continue
            cross_arch.append({
                "target": target, "feature_set": fset,
                "splittype": h_sel["splittype"],
                "headline": {"ranking": [list(t) for t in h["ranking"]],
                             "arch_family": h["arch_family"], "metric": h["metric"]},
                "runner": {"ranking": [list(t) for t in r["ranking"]],
                           "arch_family": r["arch_family"], "metric": r["metric"]},
            })

    park = []
    for sel in selections:
        if sel["target"] != "migraine" or sel["feature_set"] != "park_features":
            continue
        ex = parse_explain(latest_explain(sel["leaf_dir"]))
        if ex is None:
            continue
        shared, or_rank, shap_rank, rho, p = _park_ranks(ex)
        if rho is None:        # the scatter needs >= 3 shared triggers
            continue
        park.append({
            "role": sel["role"], "family": sel["family"],
            "splittype": sel["splittype"], "architecture": sel["architecture"],
            "shared": shared, "or_rank": or_rank, "shap_rank": shap_rank,
            "rho": rho, "p": p,
        })

    # Headline model per target on the deployable cell (full_features / chrono),
    # with its parsed ranking, leaf_dir, and AUROC CI - the data the paper's
    # feature-attribution and SHAP-beeswarm figures (fig_h*) read. The beeswarm
    # additionally needs the leaf's frozen shap_matrix_*.npz, found via leaf_dir.
    headline_explain = []
    for sel in selections:
        if (sel["role"] != "headline" or sel["feature_set"] != "full_features"
                or sel["splittype"] != "chrono"):
            continue
        ex = parse_explain(latest_explain(sel["leaf_dir"]))
        if ex is None:
            continue
        headline_explain.append({
            "target": sel["target"], "family": sel["family"],
            "architecture": sel["architecture"], "leaf_dir": str(sel["leaf_dir"]),
            "metric": ex["metric"],
            "ranking": [list(t) for t in ex["ranking"]],
            "auroc_lo": sel.get("auroc_lo"), "auroc_hi": sel.get("auroc_hi"),
        })

    return {"headlines": headlines, "cross_arch": cross_arch, "park": park,
            "headline_explain": headline_explain}


def _single_ranking_table(ex: dict, top_n: int = 10) -> str:
    """HTML table of one leaf's top-N feature ranking (no comparison)."""
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{f}</td><td>{v:.4f}</td></tr>"
        for i, (f, v) in enumerate(ex["ranking"][:top_n]))
    return (f"<table><tr><th>rank</th><th>feature ({ex['arch_family']})</th>"
            f"<th>{ex['metric']}</th></tr>{rows}</table>")


def _chance_caveat(sel: dict) -> str:
    """Warn when a leaf's 95% AUROC CI reaches chance (lower bound <= 0.5).

    Such a leaf is not significantly better than random, so its SHAP
    attributions explain a near-chance decision surface and should be read as
    descriptive only - the same threshold the discrimination figure hatches.
    """
    if sel is None or sel.get("auroc_lo") is None or sel["auroc_lo"] > 0.5:
        return ""
    return ("<p class='warn'>This cell's headline is not significantly above "
            f"chance (AUROC 95% CI [{sel['auroc_lo']:.3f}-{sel['auroc_hi']:.3f}] "
            "includes 0.5); its attributions describe a near-chance model and "
            "are not evidence of a real effect.</p>")


def _cell_block(target, fset, roles) -> tuple[str, bool]:
    """Render one (target, feature_set) cell within a split section.

    Returns ``(html, is_cross_family)``. Shows the cross-family ranking
    comparison when both roles have insight artefacts; a labelled solo
    ranking when only one does; a true miss only when neither does. A cell
    whose headline AUROC CI reaches chance carries a caveat, since its SHAP
    explains a near-random decision surface.
    """
    h_sel, r_sel = roles.get("headline"), roles.get("runner_up")
    h = parse_explain(latest_explain(h_sel["leaf_dir"])) if h_sel else None
    r = parse_explain(latest_explain(r_sel["leaf_dir"])) if r_sel else None
    head = f"<h3>{target} / {fset}</h3>"
    if h is not None and r is not None:
        fams = {h["arch_family"], r["arch_family"]}
        cross = "xgboost" in fams and bool({"tabpfn", "autotabpfn"} & fams)
        figs = (_leaf_figures(h_sel, "headline")
                + _leaf_figures(r_sel, "runner-up"))
        crossfig = ""
        # Only a genuine cross-architecture pair earns the comparison figure:
        # different families and different leaves. Same-leaf / same-family
        # pairs would draw a model against itself (identical bars).
        if cross and h_sel["leaf_dir"] != r_sel["leaf_dir"]:
            _FIG_DIR.mkdir(exist_ok=True)
            out_png = (_FIG_DIR
                       / f"crossarch_{target}_{fset}_{h_sel['splittype']}.png")
            if cross_arch_figure(h, r, h_sel, out_png) is not None:
                crossfig = (
                    "<div style='margin:0.6rem 0'>"
                    "<div style='font-size:0.82rem;color:#555'>Cross-architecture "
                    "attribution: each model's top features as a share of its "
                    "own total mean |SHAP| (headline vs runner-up). Normalised "
                    "because the two explainers' absolute magnitudes are not "
                    "comparable; rank agreement is in the table above.</div>"
                    + _embed_png(out_png, max_width=560) + "</div>")
        return (head + _chance_caveat(h_sel)
                + f"<p>headline: {_leaf_meta(h_sel)}<br>runner-up: "
                  f"{_leaf_meta(r_sel)}</p>" + _rank_overlap_table(h, r)
                + crossfig + f"<div>{figs}</div>"), cross
    if h is not None or r is not None:
        sel = h_sel if h is not None else r_sel
        ex = h if h is not None else r
        return (head + _chance_caveat(sel)
                + f"<p>Only one architecture insighted in this cell: "
                f"{_leaf_meta(sel)}. No cross-family comparison available.</p>"
                + _single_ranking_table(ex)
                + f"<div>{_leaf_figures(sel, 'leaf')}</div>"), False
    return (head + "<p class='warn'>No insight artefacts on disk for this "
            "cell.</p>"), False


def build_comparison(selections: list[dict]) -> tuple[str, bool]:
    """Build the headline-vs-runner-up ranking-diff HTML, categorised by
    split type (chronological / stratified / patient) so the honest result
    and the leakage/generalisation contrasts are read separately. Returns
    ``(html, has_cross_family_pair)``.
    """
    by_split: dict[str, dict[tuple, dict]] = {}
    for sel in selections:
        by_split.setdefault(sel["splittype"], {}).setdefault(
            (sel["target"], sel["feature_set"]), {})[sel["role"]] = sel

    sections = []
    has_cross_family = False
    for split_type in ("chrono", "stratified", "patient"):
        cells = by_split.get(split_type)
        if not cells:
            continue
        blocks = []
        for (target, fset), roles in sorted(cells.items()):
            block, cross = _cell_block(target, fset, roles)
            has_cross_family = has_cross_family or cross
            blocks.append(block)
        label = SPLIT_LABELS.get(split_type, split_type)
        sections.append(f"<h2>{label}</h2>\n" + "\n".join(blocks))
    return "\n".join(sections), has_cross_family


def build_summary_table(selections: list[dict]) -> str:
    """Per-cell best model (headline) in each split type, so the reader sees
    which architecture/HP wins per cell and how the honest chronological
    result compares to the leakage-inflated stratified one and the
    patient-generalisation one. Reports full provenance per entry.
    """
    head = {(s["target"], s["feature_set"], s["splittype"]): s
            for s in selections if s["role"] == "headline"}
    cells = sorted({(s["target"], s["feature_set"]) for s in selections})

    def cell_entry(s):
        if s is None:
            return "<td style='color:#999'>-</td>"
        hp = s["hp_strategy"] or "NonHP"
        calib = f"{s['calib_slope']:+.2f}" if s.get("calib_slope") is not None else "n/a"
        return (f"<td>{s['family']} {s['datasplit']} <i>{hp}</i><br>"
                f"AUROC {s['auroc_mean']:.3f}, calib {calib}</td>")

    rows = []
    for (t, fs) in cells:
        tds = "".join(cell_entry(head.get((t, fs, st)))
                      for st in ("chrono", "stratified", "patient"))
        rows.append(f"<tr><td><b>{t}</b><br>{fs}</td>{tds}</tr>")
    return ("<h2>Best model per cell and split (headline)</h2>"
            "<p class='note'>The <b>chronological</b> column is the deployable, "
            "forecasting-honest best; <b>stratified</b> is not deployable - a "
            "random shuffle leaks adjacent-day signal through history features "
            "across the train/test boundary, so it is excluded by construction "
            "regardless of its measured AUROC; <b>patient</b> is generalisation "
            "to unseen patients. AUROC is hold-out test.</p>"
            "<table><tr><th>target / feature set</th><th>chronological "
            "(honest)</th><th>stratified (leaky)</th><th>patient "
            "(generalisation)</th></tr>" + "".join(rows) + "</table>")


def build_key_results(headlines: list[dict]) -> str:
    """A concise, data-driven key-results box for a paper reader.

    Every figure is computed from the current sweep, so the summary stays
    accurate as coverage changes. It states only CI-defensible claims: the
    best deployable (chronological) forecast and its AUPRC lift, the
    architecture and tuning win counts, the leakage verdict by CI separation,
    and the cells not significantly above chance.
    """
    heads = [s for s in headlines if s.get("role") == "headline"]
    if not heads:
        return ""
    chrono = [s for s in heads if s["splittype"] == "chrono"
              and s.get("auroc_lo") is not None and s["auroc_lo"] > 0.5]
    bits = []
    if chrono:
        best = max(chrono, key=lambda s: s["auroc_mean"])
        prev = _test_prevalence(best["target"], best["datasplit"],
                                best["splittype"])
        lift = (f", AUPRC lift {best['auprc_mean'] / prev:.1f}x"
                if prev and best.get("auprc_mean") else "")
        bits.append(
            f"<li><b>Best deployable (chronological) forecast:</b> "
            f"{best['target']} / {best['feature_set'].replace('_features','')} "
            f"({best['family']}, {best['datasplit']}), AUROC "
            f"{best['auroc_mean']:.3f} [{best['auroc_lo']:.3f}-"
            f"{best['auroc_hi']:.3f}]{lift}.</li>")
    fam = {}
    hp_nonhp = 0
    for s in heads:
        fam[s["family"]] = fam.get(s["family"], 0) + 1
        if not s.get("hp_strategy"):
            hp_nonhp += 1
    fam_str = ", ".join(f"{v} {k}" for k, v in sorted(fam.items(),
                                                      key=lambda kv: -kv[1]))
    bits.append(
        f"<li><b>Architecture and tuning:</b> {fam_str} across {len(heads)} "
        f"headlines; library-default (NonHP) models win {hp_nonhp} of "
        f"{len(heads)}.</li>")
    # leakage: count chronological vs stratified CI separations
    by = {}
    for s in heads:
        by.setdefault((s["target"], s["feature_set"]), {})[s["splittype"]] = s
    n_sig = sum(1 for v in by.values()
                if "chrono" in v and "stratified" in v
                and v["stratified"]["auroc_lo"] > v["chrono"]["auroc_hi"])
    bits.append(
        f"<li><b>Stratified leakage:</b> excluded by split design; in "
        f"{n_sig} of {len(by)} cells does the stratified AUROC CI separate "
        f"above chronological, so no measured inflation is claimed.</li>")
    nac = [f"{s['target']}/{s['feature_set'].replace('_features','')}/"
           f"{s['splittype']}" for s in heads
           if s.get("auroc_lo") is not None and s["auroc_lo"] <= 0.5]
    if nac:
        bits.append(
            f"<li><b>Not above chance</b> (95% CI includes 0.5): "
            f"{', '.join(nac)}.</li>")
    return ("<h2>Key results</h2><div class='note'><ul style='margin:0'>"
            + "".join(bits) + "</ul></div>")


def build_headline_composition(headlines: list[dict]) -> str:
    """Aggregate which architecture families and hyperparameter-tuning
    strategies actually win the headline, per split type.

    The per-cell table answers "what won here"; this answers "what wins
    overall". It counts the headline leaves by architecture family and by HP
    strategy (NonHP = library defaults, the TabPFN zero-shot regime and the
    untuned XGBoost stack) for each split type, so the reader sees at a glance
    whether tuning earns its keep and which family carries the benchmark.
    """
    rows = [s for s in headlines if s.get("role") == "headline"]
    if not rows:
        return ""
    splits = ("chrono", "stratified", "patient")

    def tally(field, transform):
        seen, counts = {}, {}
        for st in splits:
            counts[st] = {}
            for s in rows:
                if s["splittype"] != st:
                    continue
                key = transform(s)
                counts[st][key] = counts[st].get(key, 0) + 1
                seen[key] = True
        return list(seen), counts

    def render(title, field, transform):
        keys, counts = tally(field, transform)
        keys.sort()
        body = []
        for k in keys:
            tds = "".join(f"<td>{counts[st].get(k, 0)}</td>" for st in splits)
            total = sum(counts[st].get(k, 0) for st in splits)
            body.append(f"<tr><td>{k}</td>{tds}<td><b>{total}</b></td></tr>")
        return (f"<p class='note'>{title}</p><table><tr><th></th>"
                "<th>chronological</th><th>stratified</th><th>patient</th>"
                "<th>total</th></tr>" + "".join(body) + "</table>")

    fam = render("Headline architecture family (count of cells won per split):",
                 "family", lambda s: s["family"])
    hp = render("Headline hyperparameter-tuning strategy:",
                "hp", lambda s: (s.get("hp_strategy")
                                 + (f"/{s['hp_variant']}" if s.get("hp_variant")
                                    else "")) if s.get("hp_strategy")
                else "NonHP (library defaults)")
    return ("<h2>What wins the headline, overall</h2>" + fam + hp)


def build_split_contrast(headlines: list[dict]) -> str:
    """Per-cell chronological-vs-stratified AUROC contrast with a CI-separation
    verdict, so the leakage claim is reported only as far as the data support.

    Stratified is excluded on principle (the split design leaks), but whether
    it *measurably* inflates AUROC is an empirical question. This table gives
    the chronological and stratified headline AUROCs, their difference, and
    whether the 95% CIs separate: only a stratified interval lying entirely
    above the chronological one is evidence of inflation at this sample size.
    """
    head = {(s["target"], s["feature_set"]): {} for s in headlines
            if s.get("role") == "headline"}
    for s in headlines:
        if s.get("role") == "headline":
            head[(s["target"], s["feature_set"])][s["splittype"]] = s
    rows, n_sig = [], 0
    for (t, fs), by in sorted(head.items()):
        c, st = by.get("chrono"), by.get("stratified")
        if not c or not st:
            continue
        delta = st["auroc_mean"] - c["auroc_mean"]
        if st["auroc_lo"] > c["auroc_hi"]:
            verdict, color = "stratified higher (CIs separate)", "#b2182b"
            n_sig += 1
        elif c["auroc_lo"] > st["auroc_hi"]:
            verdict, color = "chronological higher (CIs separate)", "#1a7a3a"
        else:
            verdict, color = "not distinguishable (CIs overlap)", "#666666"
        rows.append(
            f"<tr><td>{t} / {fs.replace('_features','')}</td>"
            f"<td>{c['auroc_mean']:.3f} [{c['auroc_lo']:.3f}-{c['auroc_hi']:.3f}]</td>"
            f"<td>{st['auroc_mean']:.3f} [{st['auroc_lo']:.3f}-{st['auroc_hi']:.3f}]</td>"
            f"<td>{delta:+.3f}</td>"
            f"<td style='color:{color}'>{verdict}</td></tr>")
    if not rows:
        return ""
    verdict_line = (
        f"In {n_sig} of {len(rows)} cells the stratified CI lies entirely above "
        "the chronological one; elsewhere the difference is within sampling "
        "noise. The stratified split is excluded for its leakage mechanism, not "
        "on the strength of a measured inflation." if n_sig else
        "In no cell does the stratified CI separate from the chronological one, "
        "so the data do not establish a measurable inflation at this sample "
        "size; the stratified split is excluded for its leakage mechanism, not "
        "for an observed inflation.")
    return ("<p class='note'>" + verdict_line + "</p>"
            "<table><tr><th>cell</th><th>chronological AUROC</th>"
            "<th>stratified AUROC</th><th>&Delta;</th><th>CI verdict</th></tr>"
            + "".join(rows) + "</table>")


def build_park_check(selections: list[dict]) -> str:
    """Build the Park-OR check HTML for the migraine/park cells."""
    park = [s for s in selections
            if s["target"] == "migraine" and s["feature_set"] == "park_features"]
    blocks = []
    for sel in park:
        ex = parse_explain(latest_explain(sel["leaf_dir"]))
        if ex is None:
            continue
        shared, or_rank, shap_rank, rho, p = _park_ranks(ex)
        rows = []
        for f in sorted(PARK_TABLE4_OR, key=PARK_TABLE4_OR.get, reverse=True):
            sval = next((v for n, v in ex["ranking"] if n == f), None)
            rows.append(
                f"<tr><td>{f}</td><td>{PARK_TABLE4_OR[f]}</td>"
                f"<td>{or_rank[f]}</td>"
                f"<td>{shap_rank.get(f, '-')}</td>"
                f"<td>{f'{sval:.4f}' if sval is not None else '-'}</td></tr>")
        rho_str = (f"{rho:+.3f} (p = {p:.3f}, n = {len(shared)})"
                   if rho is not None else "n/a")
        scatter = ""
        if rho is not None:
            _FIG_DIR.mkdir(exist_ok=True)
            out_png = _FIG_DIR / f"parkrank_{sel['role']}_{sel['splittype']}.png"
            if park_scatter_figure(shared, or_rank, shap_rank, rho,
                                   sel, out_png) is not None:
                scatter = _embed_png(out_png, max_width=420)
        blocks.append(
            f"<h3>{sel['role']}: {sel['architecture']} "
            f"({ex['arch_family']}, {ex['metric']})</h3>"
            f"<p>{_leaf_meta(sel)}</p>"
            f"<p>Spearman correlation (SHAP rank vs Park OR rank): "
            f"<b>{rho_str}</b> over {len(shared)} triggers.</p>"
            f"<table><tr><th>trigger</th><th>Park OR</th><th>OR rank</th>"
            f"<th>SHAP rank</th><th>{ex['metric']}</th></tr>"
            + "".join(rows) + f"</table>{scatter}")
    return "\n".join(blocks)


_STYLE = ("<style>body{font-family:sans-serif;margin:2rem;max-width:60rem}"
          "table{border-collapse:collapse;margin:0.5rem 0}"
          "th,td{border:1px solid #ccc;padding:3px 8px;font-size:0.9rem}"
          "th{background:#f0f0f0}.warn{color:#b2182b}h3{margin-top:1.5rem}"
          ".note{background:#f6f8fa;border-left:4px solid #8895a7;"
          "padding:8px 14px;font-size:0.9rem;margin:0.6rem 0}</style>")

_PREAMBLE = (
    "<div class='note'><b>How to read this report.</b> For each "
    "(target, feature set) cell and split type, the <i>headline</i> is the leaf "
    "chosen by a multi-metric composite - AUROC and AUPRC bucketed at a 0.02 "
    "noise tolerance, then calibration slope closest to 1 as the tie-break, with "
    "degenerate-calibration leaves excluded - and the <i>runner-up</i> is the "
    "best such leaf of a different architecture family whose AUROC CI overlaps "
    "the headline (the cross-architecture contrast). SHAP is computed on the "
    "calibrated positive-class probability (KernelSHAP for the XGBoost stack, the "
    "TabPFN-native explainer for TabPFN), so calibration quality is part of the "
    "selection. All metrics are hold-out test, with the data-split ratio, split "
    "type and hyperparameter-tuning configuration shown on every leaf line."
    "<br><b>Split types.</b> <i>Chronological</i> = forecasting-honest (train on "
    "the past) - the trustworthy section. <i>Stratified</i> = random shuffle, "
    "optimistically biased because history features carry adjacent-day signal "
    "across the train/test boundary (not deployable). <i>Patient hold-out</i> = "
    "generalisation to unseen patients. Attributions are comparable within a "
    "split type.</div>")


_PARK_PREAMBLE = (
    "<div class='note'><b>What this compares.</b> Park et al. 2016 report odds "
    "ratios for self-reported migraine triggers in the SHD cohort - a "
    "population-level, same-day association between a trigger and a migraine day. "
    "The model's mean |SHAP| rank is a data-driven importance for <i>next-day</i> "
    "individual prediction from the park trigger features. The two answer "
    "different questions, so this is a convergence check, not a validation: a "
    "positive Spearman correlation means the model recovers Park's trigger "
    "ordering; a value near zero or negative means it weights the triggers "
    "differently, which is expected when same-day cross-sectional odds ratios are "
    "asked to drive a next-day forecast on a small feature set. Each panel shows "
    "the leaf's full provenance (architecture, hyperparameter-tuning, data-split "
    "ratio and split type) and the odds-ratio-rank vs SHAP-rank scatter against "
    "the agreement diagonal.</div>")


def main() -> int:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(__file__).resolve().parent

    # Rotate any prior comparison outputs into experiment/2/results/<date>/
    # using the shared dated-archive algorithm before writing the new ones.
    archive_previous_outputs(out_dir, out_dir / "results", COMPARE_PATTERNS)

    # Resolve each selection to a leaf that actually carries insight
    # artefacts (the insight pass and the live AUROC ranking can disagree).
    all_rows = _select.collect_holdout_rows()
    raw_selections = select_insight_leaves()
    selections = [resolve_insighted(s, all_rows) for s in raw_selections]

    # Freeze the figure-ready findings to figdata_<ts>.json (rotated into
    # results/<date>/ alongside the reports) so the docs per-figure scripts
    # render from a pinned snapshot; the report figures below use the same
    # in-memory headlines, so the two never disagree.
    figdata = _gather_figdata(raw_selections, selections)
    figdata["generated"] = ts
    figdata_path = out_dir / f"figdata_{ts}.json"
    figdata_path.write_text(json.dumps(figdata, indent=2))
    print(f"wrote {figdata_path}")
    headlines = figdata["headlines"]

    # Split-selection figure: a discrimination-performance chart reading the
    # best AUROC per split from the frozen headlines.
    _FIG_DIR.mkdir(exist_ok=True)
    split_fig = split_auroc_figure(headlines, _FIG_DIR / "split_auroc.png")
    split_block = ""
    if split_fig is not None:
        split_block = (
            "<h2>Discrimination across split types</h2>"
            "<p class='note'>Best hold-out AUROC per cell, one bar per split. "
            "The <b>chronological</b> bar is the deployable forecast; "
            "<b>stratified</b> is the leakage contrast - a random shuffle places "
            "adjacent days, which share history / rolling feature values, on both "
            "sides of the train/test boundary; <b>patient</b> is generalisation "
            "to unseen patients. The leakage is a property of the split design, "
            "so stratified is excluded by construction; the magnitude of any "
            "empirical inflation is reported separately below, since the wide CIs "
            "at this sample size do not by themselves establish it.</p>"
            + _embed_png(_FIG_DIR / "split_auroc.png", max_width=720)
            + build_split_contrast(raw_selections))
    lift_fig = auprc_lift_figure(headlines, _FIG_DIR / "auprc_lift.png")
    if lift_fig is not None:
        split_block += (
            "<p class='note'>Under the heavy class imbalance (the migraine "
            "positive rate is ~5-7 %), AUROC can look respectable while "
            "precision-recall stays near the base rate. AUPRC lift = AUPRC "
            "divided by the test-set positive prevalence puts both targets on a "
            "common scale; <b>1.0 is no skill</b> (no better than predicting the "
            "base rate). Each bar uses its own leaf's test prevalence as the "
            "baseline.</p>"
            + _embed_png(_FIG_DIR / "auprc_lift.png", max_width=720))
    calib_fig = calib_slope_figure(headlines, _FIG_DIR / "calib_slope.png")
    if calib_fig is not None:
        split_block += (
            "<p class='note'>Calibration is the second axis of forecast "
            "quality: discrimination ranks days, calibration scales the "
            "probabilities. The selection prefers a slope near 1.0 and excludes "
            "the shaded (inverted or mis-scaled) zones, so a cell sitting far "
            "from 1.0 discriminates without yielding trustworthy "
            "probabilities.</p>"
            + _embed_png(_FIG_DIR / "calib_slope.png", max_width=720))

    comp_html, has_cross = build_comparison(selections)
    comp_path = out_dir / f"comparison_shap_{ts}.html"
    comp_path.write_text(
        f"<html><head>{_STYLE}</head><body>"
        f"<h1>Addition 2: cross-architecture SHAP comparison</h1>"
        f"{_PREAMBLE}"
        f"{build_key_results(raw_selections)}"
        f"<p>Cross-family (XGBoost vs TabPFN) pair present: "
        f"<b>{'yes' if has_cross else 'NO'}</b>.</p>"
        f"{build_summary_table(selections)}"
        f"{build_headline_composition(raw_selections)}"
        f"{split_block}"
        f"{comp_html}</body></html>")
    print(f"wrote {comp_path}")
    if not has_cross:
        print("WARNING: no XGBoost-vs-TabPFN cross-family pair in the "
              "selected cells; the benchmark's core comparison is missing.")

    park_html = build_park_check(selections)
    park_path = out_dir / f"park_or_check_{ts}.html"
    park_path.write_text(
        f"<html><head>{_STYLE}</head><body>"
        f"<h1>Park 2016 Table-4 OR vs mean |SHAP| rank (migraine/park)</h1>"
        f"{_PARK_PREAMBLE}"
        f"{park_html or '<p class=warn>No migraine/park insight artefacts found.</p>'}"
        f"</body></html>")
    print(f"wrote {park_path}")

    # Emit a shareable PDF alongside each HTML report (best-effort).
    html_to_pdf(comp_path)
    html_to_pdf(park_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
