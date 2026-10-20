"""BibTeX exporter for scientific papers and citations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List
from helixmind.papers.models import ClinicalStudy, Paper


class BibTeXExporter:
    """Generates standard BibTeX (.bib) files for reference managers like Zotero or Mendeley."""

    @staticmethod
    def clean_bibtex_key(text: str) -> str:
        """Create clean alphanumeric key."""
        return re.sub(r"[^a-zA-Z0-9_]+", "", text.strip().replace(" ", "_"))

    @classmethod
    def export_bibtex(
        cls,
        papers: List[Paper],
        clinical_trials: List[ClinicalStudy],
        filepath: Path,
    ) -> Path:
        """Serialize papers and trials into a valid BibTeX file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        entries: List[str] = []

        for p in papers:
            first_author = p.authors[0].split()[0] if p.authors else "HelixMind"
            year = str(p.pub_year or "2026")
            clean_first = cls.clean_bibtex_key(first_author)
            clean_id = cls.clean_bibtex_key(p.pmid or p.id)
            cite_key = f"{clean_first}{year}_{clean_id}"

            authors_fmt = " and ".join(p.authors) if p.authors else "HelixMind Consortium"
            title_clean = p.title.replace("{", "").replace("}", "")

            fields = [
                f"  author    = {{{authors_fmt}}}",
                f"  title     = {{{{{title_clean}}}}}",
                f"  year      = {{{year}}}",
            ]
            if p.journal:
                fields.append(f"  journal   = {{{p.journal}}}")
            if p.doi:
                fields.append(f"  doi       = {{{p.doi}}}")
            if p.pmid:
                fields.append(f"  note      = {{PMID: {p.pmid}}}")

            entry_str = f"@article{{{cite_key},\n" + ",\n".join(fields) + "\n}"
            entries.append(entry_str)

        for t in clinical_trials:
            key = f"ClinicalTrial_{t.nct_id}"
            title_clean = t.title.replace("{", "").replace("}", "")
            fields = [
                f"  title     = {{{{{title_clean}}}}}",
                f"  howpublished = {{\\url{{https://clinicaltrials.gov/study/{t.nct_id}}}}}",
                f"  year      = {{{2026}}}",
                f"  note      = {{Status: {t.status}, Phase: {t.phase or 'N/A'}}}",
            ]
            entry_str = f"@misc{{{key},\n" + ",\n".join(fields) + "\n}"
            entries.append(entry_str)

        content = "%\n% BibTeX bibliography exported by HelixMind Autonomous Scientific Platform\n%\n\n" + "\n\n".join(entries)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath
