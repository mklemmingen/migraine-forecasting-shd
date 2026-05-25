"""Tier-0 figure lint: deterministic checks of every fig_*.py against the
mechanical contracts in experiment/_style.py and docs/figure_design_requirements.md.

This is the free pre-filter of the figure-review pipeline: it catches the rule
violations a linter can see (inline hex, bypassing _style, non-column widths,
multi-panel without panel labels, hand-rolled reference lines) so the multimodal
review committee spends its judgement only on visual/semantic quality.

Usage: python _audit_figures.py            (lints all fig_*.py here)
Exit code is the number of FAILs (0 = clean). Not a figure; name underscored so
the figure runner's fig_*.py glob ignores it.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEX = re.compile(r"#[0-9A-Fa-f]{6}")
# figures whose categorical heatmap/grid legitimately needs discrete fills not in
# the qualitative palette (guide Section 7); inline hex there is WARN, not FAIL,
# but should still migrate to named _style constants.
HEATMAP_LIKE = {"fig_a3_coverage", "fig_a5_splits"}


def _check(path: Path) -> dict:
    src = path.read_text()
    name = path.stem
    findings = []  # (severity, code, message)

    if "import _style as S" not in src:
        findings.append(("FAIL", "import", "does not `import _style as S`"))
    if "S.apply()" not in src:
        findings.append(("FAIL", "apply", "never calls S.apply()"))
    if "S.save(" not in src:
        findings.append(("FAIL", "save", "does not save via S.save() (PDF+PNG contract)"))
    if re.search(r"\b(fig|plt)\.savefig\(", src):
        findings.append(("FAIL", "raw-save", "calls savefig() directly, bypassing S.save()"))

    # inline hex (allow the ones inside string of a comment? keep simple: any code hex)
    hexes = sorted(set(HEX.findall(src)))
    if hexes:
        sev = "WARN" if name in HEATMAP_LIKE else "FAIL"
        findings.append((sev, "inline-hex",
                         f"{len(hexes)} inline hex literal(s) {hexes[:6]} - use S.* colours"))

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

    return {"name": name, "findings": findings}


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
        print(f"{status:<8}{extra:<10} {r['name']}")
        for sev, code, msg in r["findings"]:
            print(f"         {sev:<5} [{code}] {msg}")
    print("=" * 64)
    print(f"{fails} FAIL(s) across {len(figs)} figures")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
