"""NCBI E-utilities client for PubMed and PMC retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx

from helixmind.config import Settings, get_settings
from helixmind.papers.models import Paper


class NCBIClient:
    """Client for PubMed and PMC via NCBI E-utilities."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.ncbi_base_url.rstrip("/")
        self.api_key = self.settings.ncbi_api_key
        self.tool = self.settings.ncbi_tool
        self.email = self.settings.ncbi_email
        self.timeout = 30.0

    def _default_params(self) -> Dict[str, str]:
        params = {"tool": self.tool, "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @staticmethod
    def _clean_pubmed_abstract(raw: str) -> str:
        """Strip journal citation line and author affiliation blocks from PubMed plain text."""
        if not raw:
            return ""
        if "author information" in raw.lower():
            parts = raw.split("\n\n")
            for p_idx, part in enumerate(parts):
                if "author information" in part.lower():
                    body = "\n\n".join(parts[p_idx + 1:]).strip()
                    if body:
                        return body
        return raw.strip()

    # -------------------------------------------------------------------------
    # PubMed Search (ESearch)
    # -------------------------------------------------------------------------

    async def search_pubmed_async(
        self,
        term: str,
        limit: int = 10,
        sort: str = "pub_date",
    ) -> List[str]:
        """Search PubMed for a given query term and return a list of PMIDs."""
        params = {
            **self._default_params(),
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": str(limit),
            "sort": sort,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(f"{self.base_url}/esearch.fcgi", params=params)
            res.raise_for_status()
            data = res.json()
            return data.get("esearchresult", {}).get("idlist", [])

    def search_pubmed(
        self,
        term: str,
        limit: int = 10,
        sort: str = "pub_date",
    ) -> List[str]:
        """Synchronously search PubMed."""
        params = {
            **self._default_params(),
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": str(limit),
            "sort": sort,
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.base_url}/esearch.fcgi", params=params)
            res.raise_for_status()
            data = res.json()
            return data.get("esearchresult", {}).get("idlist", [])

    # -------------------------------------------------------------------------
    # PubMed Summaries (ESummary) & Abstracts (EFetch)
    # -------------------------------------------------------------------------

    async def fetch_papers_async(self, pmids: List[str]) -> List[Paper]:
        """Fetch structured metadata and abstracts for given PMIDs."""
        if not pmids:
            return []

        # 1. Fetch metadata summaries via ESummary
        summary_params = {
            **self._default_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
            "version": "2.0",
        }

        papers: List[Paper] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            sum_res = await client.get(f"{self.base_url}/esummary.fcgi", params=summary_params)
            sum_res.raise_for_status()
            sum_data = sum_res.json().get("result", {})

            for pmid in pmids:
                item = sum_data.get(pmid)
                if not item:
                    continue

                title = item.get("title", "").rstrip(".")
                journal = item.get("source", "")
                authors = [a.get("name", "") for a in item.get("authors", []) if "name" in a]
                pub_year = None
                pubdate = item.get("pubdate", "")
                if pubdate and len(pubdate) >= 4 and pubdate[:4].isdigit():
                    pub_year = int(pubdate[:4])

                doi = None
                for article_id in item.get("articleids", []):
                    if article_id.get("idtype") == "doi":
                        doi = article_id.get("value")

                # Fetch individual abstract text via EFetch
                abstract_text = ""
                try:
                    fetch_params = {
                        **self._default_params(),
                        "db": "pubmed",
                        "id": pmid,
                        "rettype": "abstract",
                        "retmode": "text",
                    }
                    fetch_res = await client.get(f"{self.base_url}/efetch.fcgi", params=fetch_params)
                    if fetch_res.status_code == 200:
                        abstract_text = self._clean_pubmed_abstract(fetch_res.text)
                except Exception:
                    abstract_text = title

                pmc_val = item.get("pmcrefcount", 0)
                pmc_count = int(pmc_val) if str(pmc_val).isdigit() else 0

                paper = Paper(
                    id=pmid,
                    source="pubmed",
                    pmid=pmid,
                    title=title,
                    abstract=abstract_text or title,
                    authors=authors,
                    journal=journal,
                    pub_year=pub_year,
                    doi=doi,
                    is_open_access=bool(pmc_count > 0),
                    cited_by_count=pmc_count,
                )
                papers.append(paper)

        return papers

    def fetch_papers(self, pmids: List[str]) -> List[Paper]:
        """Synchronously fetch structured metadata and abstracts."""
        if not pmids:
            return []

        summary_params = {
            **self._default_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
            "version": "2.0",
        }

        papers: List[Paper] = []
        with httpx.Client(timeout=self.timeout) as client:
            sum_res = client.get(f"{self.base_url}/esummary.fcgi", params=summary_params)
            sum_res.raise_for_status()
            sum_data = sum_res.json().get("result", {})

            for pmid in pmids:
                item = sum_data.get(pmid)
                if not item:
                    continue

                title = item.get("title", "").rstrip(".")
                journal = item.get("source", "")
                authors = [a.get("name", "") for a in item.get("authors", []) if "name" in a]
                pub_year = None
                pubdate = item.get("pubdate", "")
                if pubdate and len(pubdate) >= 4 and pubdate[:4].isdigit():
                    pub_year = int(pubdate[:4])

                doi = None
                for article_id in item.get("articleids", []):
                    if article_id.get("idtype") == "doi":
                        doi = article_id.get("value")

                abstract_text = ""
                try:
                    fetch_params = {
                        **self._default_params(),
                        "db": "pubmed",
                        "id": pmid,
                        "rettype": "abstract",
                        "retmode": "text",
                    }
                    fetch_res = client.get(f"{self.base_url}/efetch.fcgi", params=fetch_params)
                    if fetch_res.status_code == 200:
                        abstract_text = self._clean_pubmed_abstract(fetch_res.text)
                except Exception:
                    abstract_text = title

                pmc_val = item.get("pmcrefcount", 0)
                pmc_count = int(pmc_val) if str(pmc_val).isdigit() else 0

                paper = Paper(
                    id=pmid,
                    source="pubmed",
                    pmid=pmid,
                    title=title,
                    abstract=abstract_text or title,
                    authors=authors,
                    journal=journal,
                    pub_year=pub_year,
                    doi=doi,
                    is_open_access=bool(pmc_count > 0),
                    cited_by_count=pmc_count,
                )
                papers.append(paper)

        return papers
