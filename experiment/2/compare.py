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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from _eval._figstyle import apply_journal_style, save_journal_figure  # noqa: E402

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


SPLIT_FIG_COLORS = {
    "chrono": "#0072B2",      # honest forecasting baseline (blue)
    "stratified": "#D55E00",  # leakage-inflated contrast (vermillion)
    "patient": "#009E73",     # generalisation to unseen patients (green)
}
SPLIT_FIG_LABELS = {
    "chrono": "chronological (honest)",
    "stratified": "stratified (leaky)",
    "patient": "patient (generalisation)",
}


def _split_auroc_figure(headlines: list[dict], out_png) -> Path | None:
    """Grouped horizontal bar of the best (headline) hold-out AUROC per
    (target, feature_set) cell, one bar per split type with 95% CI whiskers.

    This is the split-selection figure: the chronological bar is the
    deployable forecast, the stratified bar exposes the leakage inflation
    carried by history / rolling features, and the patient bar is
    generalisation to unseen patients. Feature sets without history features
    (no_rolling) show little chrono-to-stratified gap, which is the visual
    signature that the inflation is leakage rather than genuine skill.
    """
    cells, by_cell = [], {}
    for s in headlines:
        if s.get("role") != "headline":
            continue
        key = (s["target"], s["feature_set"])
        if key not in by_cell:
            by_cell[key] = {}
            cells.append(key)
        by_cell[key][s["splittype"]] = s
    if not cells:
        return None
    cells.sort()
    apply_journal_style()
    splits = ("chrono", "stratified", "patient")
    n, g = len(cells), len(splits)
    bh = 0.8 / g
    base = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * n * g / 2 + 1.4))
    for gi, st in enumerate(splits):
        ys, vals, los, his = [], [], [], []
        for ci, key in enumerate(cells):
            s = by_cell[key].get(st)
            if s is None:
                continue
            ys.append(base[ci] + (g / 2 - gi - 0.5) * bh)
            vals.append(s["auroc_mean"])
            los.append(s["auroc_mean"] - s["auroc_lo"])
            his.append(s["auroc_hi"] - s["auroc_mean"])
        ax.barh(ys, vals, height=bh, color=SPLIT_FIG_COLORS[st],
                xerr=[los, his], error_kw={"elinewidth": 0.8, "capsize": 2},
                label=SPLIT_FIG_LABELS[st])
    ax.axvline(0.5, color="#444444", lw=0.9, ls=":", zorder=0)
    ax.set_yticks(base)
    ax.set_yticklabels([f"{t}\n{fs.replace('_features','')}" for t, fs in cells],
                       fontsize=8)
    ax.set_xlim(0.45, max(0.95, max(s["auroc_hi"] for c in by_cell.values()
                                    for s in c.values()) + 0.03))
    ax.set_xlabel("best hold-out AUROC (95% CI); dotted line = chance (0.5)")
    ax.set_title("Discrimination by split type, per cell (headline model)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    save_journal_figure(fig, out_png)
    plt.close(fig)
    return out_png


def _calib_slope_figure(headlines: list[dict], out_png) -> Path | None:
    """Dot plot of the headline calibration slope per cell, one marker per
    split type, against the perfect-calibration line at 1.0 with the
    degenerate zones (<= 0 inverted, > 5 mis-scaled) shaded.

    Calibration is the second axis of forecast quality: discrimination ranks
    days, calibration scales the probabilities. The selection prefers a slope
    near 1.0 and excludes the shaded zones, so this chart shows how
    trustworthy each cell's headline probabilities are.
    """
    cells, by_cell = [], {}
    for s in headlines:
        if s.get("role") != "headline" or s.get("calib_slope") is None:
            continue
        key = (s["target"], s["feature_set"])
        if key not in by_cell:
            by_cell[key] = {}
            cells.append(key)
        by_cell[key][s["splittype"]] = s["calib_slope"]
    if not cells:
        return None
    cells.sort()
    apply_journal_style()
    splits = ("chrono", "stratified", "patient")
    n = len(cells)
    base = np.arange(n)[::-1]
    vals = [v for c in by_cell.values() for v in c.values()]
    xmax = max(2.2, max(vals) + 0.3)
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * n + 1.4))
    ax.axvspan(xmax * -0.02, 0.0, color="#D55E00", alpha=0.10, zorder=0)
    ax.axvspan(5.0, xmax, color="#D55E00", alpha=0.10, zorder=0)
    ax.axvline(1.0, color="#444444", lw=1.0, ls="--", zorder=1,
               label="perfect calibration (1.0)")
    for gi, st in enumerate(splits):
        ys = [base[ci] + (1 - gi) * 0.18 for ci, k in enumerate(cells)
              if st in by_cell[k]]
        xs = [by_cell[k][st] for k in cells if st in by_cell[k]]
        ax.scatter(xs, ys, s=55, color=SPLIT_FIG_COLORS[st], zorder=3,
                   edgecolor="white", label=SPLIT_FIG_LABELS[st])
    ax.set_yticks(base)
    ax.set_yticklabels([f"{t}\n{fs.replace('_features','')}" for t, fs in cells],
                       fontsize=8)
    ax.set_xlim(xmax * -0.02, xmax)
    ax.set_xlabel("calibration slope (1.0 = perfect; shaded zones excluded "
                  "from selection)", fontsize=9)
    ax.set_title("Calibration of the headline model, by split type",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", ncol=1, framealpha=0.95)
    save_journal_figure(fig, out_png)
    plt.close(fig)
    return out_png


def _cross_arch_figure(h, r, sel, out_png, top_n: int = 8) -> Path | None:
    """Journal-styled grouped horizontal bar of mean |SHAP| for the union of
    each model's top features, headline vs runner-up, for one cross-family
    cell. Saved as PNG + vector PDF via the shared figure style.
    """
    h_map, r_map = dict(h["ranking"]), dict(r["ranking"])
    feats: list[str] = []
    for f, _ in h["ranking"][:top_n] + r["ranking"][:top_n]:
        if f not in feats:
            feats.append(f)
    feats.sort(key=lambda f: max(h_map.get(f, 0.0), r_map.get(f, 0.0)), reverse=True)
    feats = feats[:12]
    if not feats:
        return None
    apply_journal_style()
    y = np.arange(len(feats))[::-1]
    bw = 0.4
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(feats) + 1.3))
    ax.barh(y + bw / 2, [h_map.get(f, 0.0) for f in feats], height=bw,
            color="#0072B2", label=f"headline ({h['arch_family']})")
    ax.barh(y - bw / 2, [r_map.get(f, 0.0) for f in feats], height=bw,
            color="#E69F00", label=f"runner-up ({r['arch_family']})")
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("mean |SHAP| (calibrated positive-class probability)")
    ax.set_title(f"{sel['target']} / {sel['feature_set']} - {sel['splittype']}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    save_journal_figure(fig, out_png)
    plt.close(fig)
    return out_png


def _park_scatter_figure(shared, or_rank, shap_rank, rho, sel, out_png) -> Path | None:
    """Journal-styled scatter of Park-2016 odds-ratio rank (x) against the
    model's mean |SHAP| rank (y) for the shared triggers, with the agreement
    diagonal and each trigger labelled. Points on the diagonal mean the model
    weights triggers in Park's order; the anti-diagonal (negative Spearman)
    means it inverts that order.
    """
    if len(shared) < 3:
        return None
    apply_journal_style()
    n = len(shared)
    xs = [or_rank[f] for f in shared]
    ys = [shap_rank[f] for f in shared]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([1, n], [1, n], color="#999999", lw=1.0, ls="--", zorder=1,
            label="perfect agreement")
    ax.scatter(xs, ys, s=70, color="#0072B2", zorder=3, edgecolor="white")
    for f, x, y in zip(shared, xs, ys):
        ax.annotate(f.replace("_today", ""), (x, y), fontsize=7.5,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(n + 0.5, 0.5)  # rank 1 (most important) at top
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_xlabel("Park 2016 odds-ratio rank (1 = strongest trigger)")
    ax.set_ylabel("model mean |SHAP| rank (1 = most weighted)")
    ax.set_title(f"{sel['role']} {sel['family']} - Spearman rho = {rho:+.2f}",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    save_journal_figure(fig, out_png)
    plt.close(fig)
    return out_png


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
        crossfig = ""
        # Only a genuine cross-architecture pair earns the comparison figure:
        # different families and different leaves. Same-leaf / same-family
        # pairs would draw a model against itself (identical bars).
        if cross and h_sel["leaf_dir"] != r_sel["leaf_dir"]:
            _FIG_DIR.mkdir(exist_ok=True)
            out_png = (_FIG_DIR
                       / f"crossarch_{target}_{fset}_{h_sel['splittype']}.png")
            if _cross_arch_figure(h, r, h_sel, out_png) is not None:
                crossfig = (
                    "<div style='margin:0.6rem 0'>"
                    "<div style='font-size:0.82rem;color:#555'>Cross-architecture "
                    "attribution: mean |SHAP| of each model's top features "
                    "(headline vs runner-up), same calibrated-probability "
                    "target.</div>"
                    + _embed_png(out_png, max_width=560) + "</div>")
        return (head
                + f"<p>headline: {_leaf_meta(h_sel)}<br>runner-up: "
                  f"{_leaf_meta(r_sel)}</p>" + _rank_overlap_table(h, r)
                + crossfig + f"<div>{figs}</div>"), cross
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
        scatter = ""
        if rho is not None:
            _FIG_DIR.mkdir(exist_ok=True)
            out_png = _FIG_DIR / f"parkrank_{sel['role']}_{sel['splittype']}.png"
            if _park_scatter_figure(shared, or_rank, shap_rank, rho,
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

    # Split-selection figure from the pre-resolve headlines: this is a
    # discrimination-performance chart, so it reads the true best AUROC per
    # split regardless of which leaf has been insighted yet.
    _FIG_DIR.mkdir(exist_ok=True)
    split_fig = _split_auroc_figure(raw_selections, _FIG_DIR / "split_auroc.png")
    split_block = ""
    if split_fig is not None:
        split_block = (
            "<h2>Discrimination across split types</h2>"
            "<p class='note'>Best hold-out AUROC per cell, one bar per split. "
            "The <b>chronological</b> bar is the deployable forecast; "
            "<b>stratified</b> exposes the optimistic inflation that history / "
            "rolling features leak across a random train/test boundary; "
            "<b>patient</b> is generalisation to unseen patients. A small "
            "chronological-to-stratified gap for the no-rolling feature set is "
            "the signature that the stratified inflation is leakage, not "
            "skill.</p>" + _embed_png(_FIG_DIR / "split_auroc.png", max_width=720))
    calib_fig = _calib_slope_figure(raw_selections, _FIG_DIR / "calib_slope.png")
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
        f"<p>Cross-family (XGBoost vs TabPFN) pair present: "
        f"<b>{'yes' if has_cross else 'NO'}</b>.</p>"
        f"{build_summary_table(selections)}"
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
