"""LaTeX (.tex) academic article generator formatted for peer-reviewed journals."""

from __future__ import annotations

import re
from pathlib import Path


class LaTeXExporter:
    """Exports scientific Markdown investigations to formal academic LaTeX articles."""

    @staticmethod
    def escape_latex(text: str) -> str:
        """Escape LaTeX special characters."""
        text = text.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
        text = text.replace("_", "\\_").replace("#", "\\#")
        return text

    @classmethod
    def export_latex(
        cls,
        markdown_content: str,
        filepath: Path,
    ) -> Path:
        """Convert GitHub Markdown paper into a structured LaTeX source file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        lines = markdown_content.split("\n")
        title = "Autonomous Scientific Investigation"
        body_lines = []
        in_abstract = False
        abstract_lines = []

        for line in lines:
            if line.startswith("# ") and title == "Autonomous Scientific Investigation":
                title = line[2:].strip()
                continue
            if line.startswith("## Abstract"):
                in_abstract = True
                continue
            if in_abstract:
                if line.startswith("## ") or line.startswith("---"):
                    in_abstract = False
                else:
                    if line.strip():
                        abstract_lines.append(cls.escape_latex(line))
                    continue

            # Convert markdown section headers
            if line.startswith("## "):
                body_lines.append(f"\n\\section{{{cls.escape_latex(line[3:].strip())}}}")
            elif line.startswith("### "):
                body_lines.append(f"\n\\subsection{{{cls.escape_latex(line[4:].strip())}}}")
            elif line.startswith("#### "):
                body_lines.append(f"\n\\subsubsection{{{cls.escape_latex(line[5:].strip())}}}")
            elif line.startswith("- "):
                body_lines.append(f"\\item {cls.escape_latex(line[2:].strip())}")
            elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. "):
                body_lines.append(f"\\item {cls.escape_latex(line[3:].strip())}")
            elif line.startswith("|") and "---" not in line:
                # Basic table cell escape
                cells = [cls.escape_latex(c.strip()) for c in line.split("|") if c.strip()]
                body_lines.append(" & ".join(cells) + " \\\\")
            elif line.startswith("```"):
                continue  # skip code fences in raw latex conversion
            elif line.strip():
                # Paragraph
                body_lines.append(f"\n{cls.escape_latex(line)}\n")

        latex_doc = f"""\\documentclass[11pt,a4paper,twocolumn]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[margin=2cm]{{geometry}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{microtype}}
\\usepackage{{xcolor}}

\\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=teal
}}

\\title{{\\textbf{{{cls.escape_latex(title)}}}}}
\\author{{HelixMind Autonomous Scientific Consortium\\\\
\\small \\textit{{Autonomous Multi-Agent Biomedical Intelligence}}}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{' '.join(abstract_lines)}
\\end{{abstract}}

\\noindent\\textbf{{Keywords:}} systems biology, targeted therapy, autonomous discovery, oncology

\\vspace{{0.5cm}}
\\hrule
\\vspace{{0.5cm}}

{''.join(body_lines)}

\\end{{document}}
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(latex_doc)

        return filepath
