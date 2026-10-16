#!/usr/bin/env python3
"""Build the plain Times New Roman report on macOS from report.html and CSVs.

Requires matplotlib, beautifulsoup4, Swift, XeLaTeX, Poppler, and installed
Times New Roman and STIX Two Math fonts.
Run from any directory: python3 scripts/build_report.py
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import tempfile

from bs4 import BeautifulSoup, NavigableString, Comment
from report_math import prepare_equations

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "docs/final-report"
MATH = {}


def normalize(text):
    # These symbols have no glyph in the installed Times New Roman. Spell them
    # out rather than silently introducing a second font in the PDF.
    substitutions = {"⁻": "-", "₁": "1", "₂": "2", "∈": " in ", "≫": " >> ",
                     "≳": ">~", "⌊": "floor(", "⌋": ")", "–": "-", "—": "-",
                     "‑": "-", "‖": "||"}
    for old, new in substitutions.items():
        text = text.replace(old, new)
    return text


def runs(node, bold=False, italic=False):
    if isinstance(node, Comment):
        return []
    if isinstance(node, NavigableString):
        return [{"text": normalize(re.sub(r"\s+", " ", str(node))), "bold": bold, "italic": italic}]
    if node.name == "br":
        return [{"text": "\n"}]
    if "math" in node.get("class", []):
        return [{"math": MATH[node["data-id"]]}]
    result = []
    for child in node.children:
        result.extend(runs(child, bold or node.name in ("b", "strong"), italic or node.name in ("i", "em")))
    return result


def clean_runs(node):
    result = runs(node)
    if result:
        if "text" in result[0]:
            result[0]["text"] = result[0]["text"].lstrip()
        if "text" in result[-1]:
            result[-1]["text"] = result[-1]["text"].rstrip()
    return result


def table_weights(table):
    text = table.get_text(" ", strip=True)
    count = len(table.find("tr").find_all(["th", "td"], recursive=False))
    if count == 8:
        return [2.25, 1.05, .7, .85, .9, .85, .95, 1.2]
    if count == 7:
        return [2.6, 1.05, .7, .7, .85, .9, 1.6]
    if count == 6:
        return [2.8, 1.05, .9, 1, 1, 1]
    if count == 5:
        return [3.5, 1, 1, 1, 1]
    if count == 4:
        return [1.1, 3.1, 1.1, 3.1]
    if count == 3:
        return [1.5, 1, .9] if "residual" in text[:140] else [2.6, 2.3, 3.1]
    return [1] * count


def blocks_from(node):
    if isinstance(node, NavigableString):
        return []
    kind = node.name
    if "equation" in node.get("class", []):
        return [{"type": "equation", "math": MATH[node["data-id"]]}]
    if kind in ("h1", "h2", "h3", "p"):
        classes = node.get("class", [])
        style = "header-meta" if "header-meta" in classes else ("meta" if "meta" in classes or "sub" in classes else kind)
        return [{"type": style, "runs": clean_runs(node)}]
    if kind == "hr":
        return [{"type": "rule"}]
    if kind == "pre":
        return [{"type": "code", "text": normalize(node.get_text())}]
    if kind == "table":
        rows = [[{"runs": clean_runs(cell), "right": "n" in cell.get("class", [])}
                 for cell in tr.find_all(["td", "th"], recursive=False)]
                for tr in node.find_all("tr")]
        caption = node.get("data-caption")
        return [{"type": "table", "rows": rows, "weights": table_weights(node), "caption": caption or ""}]
    if kind == "figure":
        image = (REPORT / node.find("img")["src"]).resolve().with_suffix(".png")
        return [{"type": "figure", "path": str(image), "caption": clean_runs(node.find("figcaption"))}]
    if kind in ("ul", "ol"):
        return [{"type": "p", "runs": clean_runs(item)} for item in node.find_all("li", recursive=False)]
    result = []
    for child in node.children:
        result.extend(blocks_from(child))
    return result


def main():
    global MATH
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-figures", action="store_true", help="Reuse the existing figures for a text/layout edit")
    args = parser.parse_args()
    if not args.skip_figures:
        subprocess.run([sys.executable, str(REPO / "scripts/report_figs.py"), "514537"], check=True)
    soup = BeautifulSoup((REPORT / "report.html").read_text(), "html.parser")
    MATH = prepare_equations(soup, REPO / "docs/equations")
    blocks = []
    for child in soup.body.children:
        blocks.extend(blocks_from(child))
    with tempfile.TemporaryDirectory(prefix="hw2-report-") as scratch:
        scratch = Path(scratch)
        payload = scratch / "document.json"
        payload.write_text(json.dumps({"blocks": blocks}, ensure_ascii=False))
        subprocess.run(["swift", "-module-cache-path", str(scratch / "swift-cache"),
                        str(REPO / "scripts/render_report.swift"), str(payload),
                        str(REPORT / "report.pdf")], check=True)


if __name__ == "__main__":
    main()
