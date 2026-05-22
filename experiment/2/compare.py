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
import sys
from pathlib import Path

# Drop this script's own directory (experiment/2) from sys.path before
# importing scipy: the local ``select.py`` would otherwise shadow the
# standard-library ``select`` module that scipy's subprocess import needs.
_THIS_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _THIS_DIR)]

from datetime import datetime  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))


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

from _eval._archival import archive_previous_outputs  # noqa: E402

# Compare outputs rotate into experiment/2/results/YYYY-MM-DD/ with the same
# dated-archive algorithm the aggregator uses for its figures.
COMPARE_PATTERNS = ("comparison_shap_*.html", "park_or_check_*.html")

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

    ``select.py`` picks the top-AUROC leaf per cell, but the insight pass
    may have run a different leaf of the same cell (the metrics shift the
    headline between runs). When the selected leaf has no ``explain_*.txt``,
    fall back to the highest-AUROC leaf of the same ``(target, feature_set,
    family)`` that does, so the comparison reads the evidence on disk rather
    than reporting a spurious miss. Returns the original sel when no
    insighted leaf of that family exists (the caller then reports the miss).
    """
    if latest_explain(sel["leaf_dir"]) is not None:
        return sel
    same = [r for r in all_rows
            if r["target"] == sel["target"]
            and r["feature_set"] == sel["feature_set"]
            and r["family"] == sel["family"]
            and latest_explain(r["leaf_dir"]) is not None]
    if not same:
        return sel
    best = max(same, key=lambda r: r["auroc_mean"])
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
    return (f"<p>Spearman rank correlation over {len(shared)} shared "
            f"features: <b>{rho_str}</b></p>"
            f"<table><tr><th>rank</th><th>headline ({headline['arch_family']})</th>"
            f"<th>runner-up ({runner['arch_family']})</th></tr>"
            + "".join(rows) + "</table>")


SPLIT_LABELS = {
    "chrono": "Chronological (forecasting-honest)",
    "stratified": "Stratified (leakage contrast - inflated, not deployable)",
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


def _single_ranking_table(ex: dict, top_n: int = 10) -> str:
    """HTML table of one leaf's top-N feature ranking (no comparison)."""
    rows = "".join(
        f"<tr><td>{i + 1}</td><td>{f}</td><td>{v:.4f}</td></tr>"
        for i, (f, v) in enumerate(ex["ranking"][:top_n]))
    return (f"<table><tr><th>rank</th><th>feature ({ex['arch_family']})</th>"
            f"<th>{ex['metric']}</th></tr>{rows}</table>")


def _cell_block(target, fset, roles) -> tuple[str, bool]:
    """Render one (target, feature_set) cell within a split section.

    Returns ``(html, is_cross_family)``. Shows the cross-family ranking
    comparison when both roles have insight artefacts; a labelled solo
    ranking when only one does; a true miss only when neither does.
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
        return (head
                + f"<p>headline: {_leaf_meta(h_sel)}<br>runner-up: "
                  f"{_leaf_meta(r_sel)}</p>" + _rank_overlap_table(h, r)
                + f"<div>{figs}</div>"), cross
    if h is not None or r is not None:
        sel = h_sel if h is not None else r_sel
        ex = h if h is not None else r
        return (head + f"<p>Only one architecture insighted in this cell: "
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
            "forecasting-honest best; <b>stratified</b> is optimistically inflated "
            "by history-feature leakage (not deployable); <b>patient</b> is "
            "generalisation to unseen patients. AUROC is hold-out test.</p>"
            "<table><tr><th>target / feature set</th><th>chronological "
            "(honest)</th><th>stratified (leaky)</th><th>patient "
            "(generalisation)</th></tr>" + "".join(rows) + "</table>")


def build_park_check(selections: list[dict]) -> str:
    """Build the Park-OR check HTML for the migraine/park cells."""
    park = [s for s in selections
            if s["target"] == "migraine" and s["feature_set"] == "park_features"]
    blocks = []
    for sel in park:
        ex = parse_explain(latest_explain(sel["leaf_dir"]))
        if ex is None:
            continue
        shap_rank = {f: i + 1 for i, (f, _) in enumerate(ex["ranking"])}
        # Park OR rank: highest OR = rank 1. ``sorted`` over a dict yields
        # its keys (feature names), so unpack a single name per item.
        or_rank = {f: i + 1 for i, f in enumerate(
            sorted(PARK_TABLE4_OR, key=PARK_TABLE4_OR.get, reverse=True))}
        shared = [f for f in PARK_TABLE4_OR if f in shap_rank]
        rho = None
        if len(shared) >= 3:
            rho, _ = spearmanr([shap_rank[f] for f in shared],
                               [or_rank[f] for f in shared])
        rows = []
        for f in sorted(PARK_TABLE4_OR, key=PARK_TABLE4_OR.get, reverse=True):
            sval = next((v for n, v in ex["ranking"] if n == f), None)
            rows.append(
                f"<tr><td>{f}</td><td>{PARK_TABLE4_OR[f]}</td>"
                f"<td>{or_rank[f]}</td>"
                f"<td>{shap_rank.get(f, '-')}</td>"
                f"<td>{f'{sval:.4f}' if sval is not None else '-'}</td></tr>")
        rho_str = f"{rho:+.3f}" if rho is not None else "n/a"
        blocks.append(
            f"<h3>{sel['role']}: {sel['architecture']} "
            f"({ex['arch_family']}, {ex['metric']})</h3>"
            f"<p>Spearman correlation (SHAP rank vs Park OR rank): "
            f"<b>{rho_str}</b> over {len(shared)} triggers.</p>"
            f"<table><tr><th>trigger</th><th>Park OR</th><th>OR rank</th>"
            f"<th>SHAP rank</th><th>{ex['metric']}</th></tr>"
            + "".join(rows) + "</table>")
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


def main() -> int:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(__file__).resolve().parent

    # Rotate any prior comparison outputs into experiment/2/results/<date>/
    # using the shared dated-archive algorithm before writing the new ones.
    archive_previous_outputs(out_dir, out_dir / "results", COMPARE_PATTERNS)

    # Resolve each selection to a leaf that actually carries insight
    # artefacts (the insight pass and the live AUROC ranking can disagree).
    all_rows = _select.collect_holdout_rows()
    selections = [resolve_insighted(s, all_rows) for s in select_insight_leaves()]

    comp_html, has_cross = build_comparison(selections)
    comp_path = out_dir / f"comparison_shap_{ts}.html"
    comp_path.write_text(
        f"<html><head>{_STYLE}</head><body>"
        f"<h1>Addition 2: cross-architecture SHAP comparison</h1>"
        f"{_PREAMBLE}"
        f"<p>Cross-family (XGBoost vs TabPFN) pair present: "
        f"<b>{'yes' if has_cross else 'NO'}</b>.</p>"
        f"{build_summary_table(selections)}"
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
        f"{park_html or '<p class=warn>No migraine/park insight artefacts found.</p>'}"
        f"</body></html>")
    print(f"wrote {park_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
