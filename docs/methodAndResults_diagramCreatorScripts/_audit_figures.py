"""Tier-0 figure lint: deterministic checks of every fig_*.py against the
mechanical contracts in experiment/_style.py and docs/figure_design_requirements.md.

This is the free pre-filter of the figure-review pipeline: it catches the rule
violations a linter can see (inline hex, bypassing _style, non-column widths,
multi-panel without panel labels, hand-rolled reference lines) so the multimodal
review committee spends its judgement only on visual/semantic quality.

It lints two layers: the fig_*.py scripts in this directory (full contract;
delegating wrappers get only the colour/raw-save guards) and the shared
experiment/_eval generators they delegate to (relaxed contract: role-palette
hex allowed, but _style routing and no named colours still enforced).

Usage: python _audit_figures.py            (lints fig_*.py here + _eval generators)
Exit code is the number of FAILs (0 = clean). Not a figure; name underscored so
the figure runner's fig_*.py glob ignores it.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The shared figure generators the fig_f*/fig_g* wrappers delegate to. They are
# linted with a relaxed ruleset (see _check_generator): they legitimately carry
# role-palette hex (ordinal ramp, family hues), so hex is allowed there, but they
# must still go through _style and never use named matplotlib colours or raw savefig.
_EXPERIMENT = HERE.parents[1] / "experiment"
GENERATOR_FILES = (
    _EXPERIMENT / "_eval" / "_benchmark_visuals.py",
    _EXPERIMENT / "_eval" / "_venn_diagrams.py",
    _EXPERIMENT / "_eval" / "_tree_diagram.py",
    _EXPERIMENT / "2" / "_figures.py",
)
HEX = re.compile(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b")   # 6- and 3-digit literals
# named matplotlib colours used as a literal colour= argument (white allowed for
# marker edges); these bypass the _style palette just like hex does.
NAMED = re.compile(r"""color\s*=\s*["'](black|grey|gray|red|green|blue|orange|"""
                   r"""purple|pink|brown|cyan|magenta|yellow|navy|teal)["']""")
# figures whose categorical heatmap/grid legitimately needs discrete fills not in
# the qualitative palette (guide Section 7); inline hex there is WARN, not FAIL,
# but should still migrate to named _style constants.
HEATMAP_LIKE = {"fig_a3_coverage", "fig_a5_splits"}

# Markers that a fig_*.py delegates to a shared generator (which owns the
# _style contract) rather than plotting inline: the _eval generators imported by
# name (fig_f*) or the Addition-2 _figures module loaded by file path (fig_g*).
# Such a wrapper, having no inline plotting, is checked one layer down, so this
# lint applies only the colour-literal and raw-save guards to it.
DELEGATION_MARKERS = ("from _benchmark_visuals import", "from _venn_diagrams import",
                      "from _tree_diagram import", "_figures.py")


def _check(path: Path) -> dict:
    src = path.read_text()
    name = path.stem
    findings = []  # (severity, code, message)

    delegated = (any(m in src for m in DELEGATION_MARKERS)
                 and "plt.subplots(" not in src)

    if not delegated:
        if "import _style as S" not in src:
            findings.append(("FAIL", "import", "does not `import _style as S`"))
        if "S.apply()" not in src:
            findings.append(("FAIL", "apply", "never calls S.apply()"))
        if "S.save(" not in src:
            findings.append(("FAIL", "save", "does not save via S.save() (PDF+PNG contract)"))
    if re.search(r"\b(fig|plt)\.savefig\(", src):
        findings.append(("FAIL", "raw-save", "calls savefig() directly, bypassing S.save()"))

    # inline hex (6- or 3-digit) and named colour= literals both bypass the palette
    hexes = sorted(set(HEX.findall(src)))
    if hexes:
        sev = "WARN" if name in HEATMAP_LIKE else "FAIL"
        findings.append((sev, "inline-hex",
                         f"{len(hexes)} inline hex literal(s) {hexes[:6]} - use S.* colours"))
    named = sorted(set(NAMED.findall(src)))
    if named:
        findings.append(("FAIL", "named-colour",
                         f"literal colour name(s) {named} - use S.* colours"))

    if delegated:
        return {"name": name, "findings": findings, "delegated": True}

    # figure width: every figsize=(W, H) should be a column width (3.5 or 7.2 in)
    for w, h in re.findall(r"figsize=\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", src):
        wf = float(w)
        if abs(wf - 3.5) > 0.3 and abs(wf - 7.2) > 0.3:
            findings.append(("FAIL", "width",
                             f"figsize width {wf}in is not a column width "
                             f"(single 3.5 / double 7.2); use S.figsize()"))
    if "S.figsize(" not in src and "figsize=(" in src:
        findings.append(("WARN", "figsize-helper", "hardcodes figsize instead of S.figsize()"))

    # multi-panel must carry panel labels
    multipanel = bool(re.search(r"subplots\(\s*[12]\s*,\s*[2-9]", src) or
                      re.search(r"subplots\(\s*[2-9]\s*,", src))
    if multipanel and "panel_label" not in src:
        findings.append(("FAIL", "panel-label",
                         "multi-panel figure without S.panel_label() on each panel"))

    # suptitle should inherit 11.5 from rcParams, not hardcode a size
    if re.search(r"suptitle\([^)]*fontsize=", src):
        findings.append(("WARN", "suptitle-size",
                         "suptitle hardcodes fontsize; rely on rcParam (11.5)"))

    # hand-rolled reference lines instead of S.refline
    if re.search(r"ax\w*\.ax[hv]line\([^)]*(black|--|':'|\"--\")", src) and "S.refline(" not in src:
        findings.append(("WARN", "refline", "hand-rolled reference line; prefer S.refline()"))

    return {"name": name, "findings": findings, "delegated": False}


def _check_generator(path: Path) -> dict:
    """Relaxed lint for an experiment/_eval figure generator.

    These modules own a palette ROLE (ordinal depth ramp, family-anchor hues,
    sequential/diverging stops), so hex literals are legitimate here and only
    reported as a count - not a FAIL. The contract they must still honour:
    route styling through ``_style`` (``apply``/``save``), never hardcode a
    named matplotlib colour, and never bypass ``save()`` with raw ``savefig``.
    """
    src = path.read_text()
    name = path.stem
    findings = []

    if "from _style import" not in src:
        findings.append(("FAIL", "import", "does not import from _style"))
    if "apply(" not in src:
        findings.append(("FAIL", "apply", "never calls apply()"))
    if "save(" not in src:
        findings.append(("FAIL", "save", "never calls save()"))
    if re.search(r"\b(fig|plt)\.savefig\(", src):
        findings.append(("FAIL", "raw-save", "calls savefig() directly, bypassing save()"))
    named = sorted(set(NAMED.findall(src)))
    if named:
        findings.append(("FAIL", "named-colour",
                         f"literal colour name(s) {named} - use S.* colours"))
    n_hex = len(set(HEX.findall(src)))
    return {"name": name, "findings": findings, "role_hex": n_hex}


def main():
    figs = sorted(HERE.glob("fig_*.py"))
    fails = 0
    print(f"Tier-0 figure lint over {len(figs)} scripts\n" + "=" * 64)
    for f in figs:
        r = _check(f)
        fa = [x for x in r["findings"] if x[0] == "FAIL"]
        wa = [x for x in r["findings"] if x[0] == "WARN"]
        fails += len(fa)
        status = "PASS" if not fa else f"FAIL({len(fa)})"
        extra = f" warn({len(wa)})" if wa else ""
        tag = " [wrapper]" if r.get("delegated") else ""
        print(f"{status:<8}{extra:<10} {r['name']}{tag}")
        for sev, code, msg in r["findings"]:
            print(f"         {sev:<5} [{code}] {msg}")

    print("-" * 64)
    print("shared figure generators (role-palette exception: hex allowed)")
    for gf in GENERATOR_FILES:
        r = _check_generator(gf)
        fa = [x for x in r["findings"] if x[0] == "FAIL"]
        fails += len(fa)
        status = "PASS" if not fa else f"FAIL({len(fa)})"
        print(f"{status:<8}{'':<10} {r['name']}  ({r['role_hex']} role-palette hex)")
        for sev, code, msg in r["findings"]:
            print(f"         {sev:<5} [{code}] {msg}")

    print("=" * 64)
    print(f"{fails} FAIL(s) across {len(figs)} figures + {len(GENERATOR_FILES)} generators")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
