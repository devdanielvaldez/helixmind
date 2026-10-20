"""PDF document parser for local laboratory manuscripts and preprints."""

from __future__ import annotations

from pathlib import Path
import re
from typing import BinaryIO, List, Optional, Union
from pypdf import PdfReader

from helixmind.config import get_settings
from helixmind.papers.models import DocumentChunk, Paper


class PDFParser:
    """Extracts text, metadata, and semantic chunks from laboratory PDF files."""

    def __init__(self, upload_dir: Optional[Path] = None):
        settings = get_settings()
        self.upload_dir = upload_dir or (settings.data_dir / "uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def parse_pdf_file(
        self,
        file_path_or_stream: Union[Path, str, BinaryIO],
        original_filename: str = "document.pdf",
    ) -> Paper:
        """Extract text content and infer title, abstract, and sections from a PDF."""
        reader = PdfReader(file_path_or_stream)

        pages_text: List[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text)
        if not full_text.strip():
            full_text = f"Unextracted graphical PDF: {original_filename}"

        # Extract title from metadata or first page
        title = ""
        meta = reader.metadata
        if meta and meta.title:
            title = meta.title.strip()

        if not title and pages_text:
            # First line of first page
            first_lines = [l.strip() for l in pages_text[0].split("\n") if len(l.strip()) > 10]
            title = first_lines[0] if first_lines else original_filename.replace(".pdf", "")

        if not title:
            title = original_filename.replace(".pdf", "")

        # Extract abstract
        abstract = ""
        abstract_match = re.search(
            r"(?i)\babstract\b[:\s\-]*([\s\S]{100,1800}?)(?:\n\s*\n\s*(?:1[\.\s]|introduction|keywords|background)|$)",
            full_text,
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        else:
            # Use first 600 characters of text
            abstract = full_text[:600].strip()

        # Extract authors
        authors = []
        if meta and meta.author:
            authors = [a.strip() for a in meta.author.split(",") if a.strip()]
        if not authors:
            authors = ["Laboratory Authors"]

        clean_id = re.sub(r"[^a-zA-Z0-9_]+", "_", original_filename.replace(".pdf", "").lower())
        paper_id = f"LOCAL_{clean_id}"

        return Paper(
            id=paper_id,
            source="local_pdf",
            title=title[:250],
            abstract=abstract,
            authors=authors,
            journal="Laboratory Manuscript / Local Archive",
            full_text=full_text,
            is_open_access=True,
        )

    def extract_chunks(self, paper: Paper, chunk_chars: int = 700) -> List[DocumentChunk]:
        """Split parsed PDF text into indexed chunks."""
        chunks: List[DocumentChunk] = []

        # 1. Abstract chunk
        chunks.append(
            DocumentChunk(
                chunk_id=f"{paper.id}_abs",
                doc_id=paper.id,
                doc_title=paper.title,
                source_type="paper",
                section="abstract",
                content=f"Title: {paper.title}\nAbstract: {paper.abstract}",
            )
        )

        # 2. Body chunks
        body = paper.full_text or ""
        paragraphs = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 10]
        cur = ""
        for i, p in enumerate(paragraphs):
            if len(cur) + len(p) < chunk_chars:
                cur += "\n" + p
            else:
                if cur.strip():
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{paper.id}_body_{len(chunks)}",
                            doc_id=paper.id,
                            doc_title=paper.title,
                            source_type="paper",
                            section="body",
                            content=cur.strip(),
                        )
                    )
                cur = p

        if cur.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{paper.id}_body_{len(chunks)}",
                    doc_id=paper.id,
                    doc_title=paper.title,
                    source_type="paper",
                    section="body",
                    content=cur.strip(),
                )
            )

        return chunks
