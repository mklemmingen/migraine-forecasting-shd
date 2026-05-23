"""HTML-to-PDF conversion for the report artefacts (shared method bank).

The Addition-2 comparison, the Addition-3 temporal report and the aggregator
all emit self-contained HTML whose figures are embedded as base64 data-URIs.
This module renders any such report to a single PDF via headless Chromium's
print-to-PDF, so each report can be shared as a PDF alongside its HTML
without a server or asset resolution.

The conversion is best-effort: a missing browser or a render failure prints a
notice and returns ``None`` rather than aborting the caller's run, so the HTML
artefacts are never blocked by the optional PDF step.

Use::

    from _eval._html_to_pdf import html_to_pdf, html_files_to_pdf
    html_to_pdf(report_path)                 # -> report_path.with_suffix('.pdf')
    html_files_to_pdf([path_a, path_b])      # -> [pdf_a, pdf_b]
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Headless print-to-PDF is the same Chromium the figure-preview screenshots
# use; no extra dependency (weasyprint / wkhtmltopdf) is introduced.
_BROWSER_CANDIDATES = (
    "chromium", "chromium-browser",
    "google-chrome", "google-chrome-stable", "chrome",
)


def find_browser() -> str | None:
    """Path to the first available Chromium/Chrome binary, or None."""
    for name in _BROWSER_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def html_to_pdf(html_path, pdf_path=None, *, timeout: int = 180) -> Path | None:
    """Render a self-contained HTML report to PDF via headless Chromium.

    ``pdf_path`` defaults to the HTML path with a ``.pdf`` suffix. Returns the
    written PDF path, or None when no browser is available or the render
    fails (a notice is printed; the caller's HTML output is unaffected).
    """
    html_path = Path(html_path).resolve()
    if not html_path.is_file():
        print(f"[pdf] source HTML not found: {html_path}")
        return None
    pdf_path = (html_path.with_suffix(".pdf") if pdf_path is None
                else Path(pdf_path)).resolve()

    browser = find_browser()
    if browser is None:
        print(f"[pdf] no Chromium/Chrome on PATH; skipped {html_path.name} "
              "(HTML still written)")
        return None

    cmd = [
        browser, "--headless", "--no-sandbox", "--disable-gpu",
        "--no-pdf-header-footer",          # drop the date/URL print header
        f"--print-to-pdf={pdf_path}",
        f"file://{html_path}",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[pdf] conversion failed for {html_path.name}: "
              f"{type(exc).__name__}: {exc}")
        return None

    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        print(f"[pdf] no PDF produced for {html_path.name}")
        return None
    print(f"[pdf] wrote {pdf_path}")
    return pdf_path


def html_files_to_pdf(paths) -> list[Path]:
    """Convert several HTML reports to PDF; returns the PDFs that succeeded."""
    out: list[Path] = []
    for p in paths:
        result = html_to_pdf(p)
        if result is not None:
            out.append(result)
    return out
