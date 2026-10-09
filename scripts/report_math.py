#!/usr/bin/env python3
"""Render the report's LaTeX equations as tightly cropped vector PDFs.

XeLaTeX uses Times New Roman for text and STIX Two Math for matching math.
The generated PNGs and CSS provide previews in the HTML report source.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile


PREAMBLE = r"""\documentclass{article}
\usepackage{amsmath,fontspec,unicode-math}
\setmainfont{TEXTFONT}
\setmathfont{MATHFONT}
\setmathrm{TEXTFONT}
\pagestyle{empty}
\hoffset=-1in
\voffset=-1in
\newbox\eqbox
\newwrite\metrics
\immediate\openout\metrics=formulas.metrics
\newcommand{\emit}[3]{%
  \setbox\eqbox=\hbox{\fontsize{#2bp}{#2bp}\selectfont$#3$}%
  \immediate\write\metrics{#1,\the\wd\eqbox,\the\ht\eqbox,\the\dp\eqbox}%
  \dimen0=\wd\eqbox\advance\dimen0 by 2pt
  \dimen2=\ht\eqbox\advance\dimen2 by\dp\eqbox\advance\dimen2 by 2pt
  \shipout\vbox{\offinterlineskip\special{papersize=\the\dimen0,\the\dimen2}%
    \kern1pt\hbox{\kern1pt\box\eqbox\kern1pt}\kern1pt}%
}
\begin{document}
"""


def prepare_equations(soup, output):
    nodes = soup.select(".math[data-latex], .equation[data-latex]")
    if not nodes:
        return {}
    output.mkdir(parents=True, exist_ok=True)
    definitions = {}
    for node in nodes:
        name = node["data-id"]
        assert name.replace("-", "").isalnum(), name
        value = (node["data-latex"], "equation" in node.get("class", []))
        assert name not in definitions or definitions[name] == value, name
        definitions[name] = value
    source = PREAMBLE.replace("TEXTFONT", os.environ.get("REPORT_TEXT_FONT", "Times New Roman"))
    source = source.replace("MATHFONT", os.environ.get("REPORT_MATH_FONT", "STIX Two Math"))
    for name, (latex, display) in definitions.items():
        prefix = r"\displaystyle " if display else ""
        source += f"\\emit{{{name}}}{{11}}{{{prefix}{latex}}}\n"
    source += "\\immediate\\closeout\\metrics\n\\end{document}\n"
    tex = output / "formulas.tex"
    metrics_file = output / "metrics.json"
    cached = (tex.exists() and tex.read_text() == source and metrics_file.exists()
              and all((output / f"{name}.{suffix}").exists()
                      for name in definitions for suffix in ("pdf", "png")))
    if not cached:
        latex_bin = shutil.which("xelatex") or "/Library/TeX/texbin/xelatex"
        with tempfile.TemporaryDirectory(prefix="hw2-latex-") as scratch:
            scratch = Path(scratch)
            (scratch / "formulas.tex").write_text(source)
            result = subprocess.run([latex_bin, "-no-shell-escape", "-interaction=nonstopmode",
                                     "-halt-on-error", "formulas.tex"], cwd=scratch,
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("XeLaTeX equation rendering failed:\n" + result.stdout[-5000:])
            if "Missing character:" in result.stdout:
                raise RuntimeError("An equation font is missing a required glyph")
            subprocess.run(["pdfseparate", str(scratch / "formulas.pdf"),
                            str(scratch / "eq-%d.pdf")], check=True)
            metadata = {}
            # TeX points have 72.27 units/inch. The report uses PDF points (72).
            factor = 72 / 72.27
            for index, line in enumerate((scratch / "formulas.metrics").read_text().splitlines(), 1):
                name, w, a, d = line.split(",")
                width, ascent, descent = [float(x.removesuffix("pt")) * factor for x in (w, a, d)]
                metadata[name] = {"width": width + 2 * factor, "ascent": ascent + factor,
                                  "descent": descent + factor, "fontSize": 11}
                shutil.copy2(scratch / f"eq-{index}.pdf", output / f"{name}.pdf")
            def preview(name):
                subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "400",
                                str(output / f"{name}.pdf"), str(output / name)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(preview, definitions))
            metrics_file.write_text(json.dumps(metadata, indent=2))
            tex.write_text(source)
    metadata = json.loads(metrics_file.read_text())
    css = []
    for name in definitions:
        m = metadata[name]
        css.append(f'[data-id="{name}"] img {{ width: {m["width"] / 11:.5f}em; '
                   f'height: {(m["ascent"] + m["descent"]) / 11:.5f}em; '
                   f'vertical-align: -{m["descent"] / 11:.5f}em; }}')
        m["path"] = str((output / f"{name}.pdf").resolve())
    (output / "math.css").write_text("\n".join(css) + "\n")
    print(f"Prepared {len(definitions)} LaTeX equations")
    return metadata
