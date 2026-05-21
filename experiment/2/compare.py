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


def build_comparison(selections: list[dict]) -> tuple[str, bool]:
    """Build the headline-vs-runner-up ranking-diff HTML.

    Returns ``(html, has_cross_family_pair)``. ``has_cross_family_pair`` is
    True when at least one cell pairs an xgboost leaf with a tabpfn /
    autotabpfn leaf (the required Addition-0-vs-Addition-1 comparison).
    """
    cells: dict[tuple, dict] = {}
    for sel in selections:
        key = (sel["target"], sel["feature_set"])
        cells.setdefault(key, {})[sel["role"]] = sel

    blocks = []
    has_cross_family = False
    for (target, fset), roles in cells.items():
        if "headline" not in roles or "runner_up" not in roles:
            continue
        h_sel, r_sel = roles["headline"], roles["runner_up"]
        h = parse_explain(latest_explain(h_sel["leaf_dir"]))
        r = parse_explain(latest_explain(r_sel["leaf_dir"]))
        if h is None or r is None:
            blocks.append(f"<h3>{target} / {fset}</h3>"
                          f"<p class='warn'>Missing insight artefact "
                          f"(headline={h is not None}, runner-up={r is not None}).</p>")
            continue
        fams = {h["arch_family"], r["arch_family"]}
        if "xgboost" in fams and ({"tabpfn", "autotabpfn"} & fams):
            has_cross_family = True
        blocks.append(
            f"<h3>{target} / {fset}</h3>"
            f"<p>headline: {h_sel['architecture']} "
            f"(AUROC {h_sel['auroc_mean']:.3f}) | runner-up: "
            f"{r_sel['architecture']} (AUROC {r_sel['auroc_mean']:.3f})</p>"
            + _rank_overlap_table(h, r))
    return "\n".join(blocks), has_cross_family


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
        # Park OR rank: highest OR = rank 1.
        or_rank = {f: i + 1 for i, (f, _) in enumerate(
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
          "th{background:#f0f0f0}.warn{color:#b2182b}h3{margin-top:1.5rem}</style>")


def main() -> int:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(__file__).resolve().parent
    selections = select_insight_leaves()

    comp_html, has_cross = build_comparison(selections)
    comp_path = out_dir / f"comparison_shap_{ts}.html"
    comp_path.write_text(
        f"<html><head>{_STYLE}</head><body>"
        f"<h1>Headline vs runner-up SHAP-ranking diff</h1>"
        f"<p>Cross-family (XGBoost vs TabPFN) pair present: "
        f"<b>{'yes' if has_cross else 'NO'}</b>.</p>"
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
