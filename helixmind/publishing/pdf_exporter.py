"""Academic PDF article generator using ReportLab."""

from __future__ import annotations

from pathlib import Path
import re
from typing import List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class PDFExporter:
    """Generates formal, publication-styled scientific PDF monographs."""

    @classmethod
    def export_pdf(
        cls,
        markdown_content: str,
        filepath: Path,
    ) -> Path:
        """Render markdown investigation paper into a formal PDF document."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=10,
        )

        meta_style = ParagraphStyle(
            "DocMeta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
            spaceAfter=12,
        )

        h1_style = ParagraphStyle(
            "SectionH1",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=14,
            spaceAfter=6,
        )

        h2_style = ParagraphStyle(
            "SectionH2",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0369a1"),
            spaceBefore=10,
            spaceAfter=4,
        )

        body_style = ParagraphStyle(
            "BodyDark",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=6,
        )

        abstract_style = ParagraphStyle(
            "AbstractText",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#0f172a"),
        )

        bullet_style = ParagraphStyle(
            "BulletStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.5,
            leftIndent=15,
            textColor=colors.HexColor("#334155"),
            spaceAfter=3,
        )

        story = []

        lines = markdown_content.split("\n")
        title = "Autonomous Scientific Investigation"
        in_abstract = False
        abstract_text = ""

        # Extract title and abstract
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
                elif line.strip():
                    abstract_text += " " + line.strip()

        # Add Title and Metadata
        story.append(Paragraph(title, title_style))
        story.append(Paragraph("<b>HelixMind Autonomous Scientific Consortium</b> | Multi-Agent Biomedical Intelligence", meta_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

        # Add Abstract Callout Box
        if abstract_text:
            abstract_flow = [
                Paragraph("<b>ABSTRACT</b>", ParagraphStyle("AbsHead", fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#0369a1"), spaceAfter=4)),
                Paragraph(abstract_text[:1200].strip(), abstract_style),
            ]
            t = Table([[abstract_flow]], colWidths=["100%"])
            t.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ])
            )
            story.append(t)
            story.append(Spacer(1, 10))

        # Parse body
        skip_abstract = False
        for line in lines:
            if line.startswith("# ") or line.startswith("## Abstract"):
                skip_abstract = True
                continue
            if skip_abstract:
                if line.startswith("## ") or line.startswith("---"):
                    skip_abstract = False
                else:
                    continue

            clean = line.strip()
            if not clean:
                continue

            # Headers
            if line.startswith("## "):
                story.append(Paragraph(line[3:].strip(), h1_style))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=6))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:].strip(), h2_style))
            elif line.startswith("#### "):
                story.append(Paragraph(f"<b>{line[5:].strip()}</b>", h2_style))
            elif line.startswith("- "):
                clean_text = line[2:].replace("**", "<b>").replace("**", "</b>")
                story.append(Paragraph(f"• {clean_text}", bullet_style))
            elif line.startswith("1. ") or line.startswith("2. ") or line.startswith("3. ") or line.startswith("4. ") or line.startswith("5. "):
                clean_text = line[3:].replace("**", "<b>").replace("**", "</b>")
                story.append(Paragraph(clean_text, bullet_style))
            elif line.startswith("```"):
                continue  # skip code fences in PDF narrative
            elif line.startswith("|") and "---" not in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) >= 2:
                    cell_text = " | ".join(cells[:4])
                    story.append(Paragraph(f"<font size=8 color='#475569'>{cell_text}</font>", bullet_style))
            else:
                formatted_body = clean.replace("**", "<b>").replace("**", "</b>")
                story.append(Paragraph(formatted_body, body_style))

        # Build PDF
        doc.build(story)
        return filepath
