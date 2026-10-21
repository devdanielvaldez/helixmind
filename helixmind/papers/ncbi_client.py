"""NCBI E-utilities client for PubMed and PMC retrieval."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET
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

    @staticmethod
    def _parse_pubmed_xml(xml_content: str) -> Dict[str, str]:
        """Extract structured abstracts from PubMed XML response keyed by PMID."""
        abstracts: Dict[str, str] = {}
        if not xml_content:
            return abstracts
        try:
            root = ET.fromstring(xml_content)
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//MedlineCitation/PMID")
                if pmid_elem is None or not pmid_elem.text:
                    continue
                pmid = pmid_elem.text.strip()
                sections = []
                for elem in article.findall(".//Abstract/AbstractText"):
                    text = "".join(elem.itertext()).strip()
                    if not text:
                        continue
                    label = elem.get("Label")
                    if label:
                        sections.append(f"{label}: {text}")
                    else:
                        sections.append(text)
                if sections:
                    abstracts[pmid] = "\n\n".join(sections).strip()

            for book in root.findall(".//PubmedBookArticle"):
                pmid_elem = book.find(".//BookDocument/PMID")
                if pmid_elem is None or not pmid_elem.text:
                    continue
                pmid = pmid_elem.text.strip()
                sections = []
                for elem in book.findall(".//Abstract/AbstractText"):
                    text = "".join(elem.itertext()).strip()
                    if not text:
                        continue
                    label = elem.get("Label")
                    if label:
                        sections.append(f"{label}: {text}")
                    else:
                        sections.append(text)
                if sections:
                    abstracts[pmid] = "\n\n".join(sections).strip()
        except Exception:
            pass
        return abstracts

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
        data = {
            **self._default_params(),
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": str(limit),
            "sort": sort,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(f"{self.base_url}/esearch.fcgi", data=data)
            res.raise_for_status()
            res_data = res.json()
            return res_data.get("esearchresult", {}).get("idlist", [])

    def search_pubmed(
        self,
        term: str,
        limit: int = 10,
        sort: str = "pub_date",
    ) -> List[str]:
        """Synchronously search PubMed."""
        data = {
            **self._default_params(),
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": str(limit),
            "sort": sort,
        }
        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(f"{self.base_url}/esearch.fcgi", data=data)
            res.raise_for_status()
            res_data = res.json()
            return res_data.get("esearchresult", {}).get("idlist", [])

    # -------------------------------------------------------------------------
    # PubMed Summaries (ESummary) & Abstracts (EFetch)
    # -------------------------------------------------------------------------

    async def fetch_papers_async(self, pmids: List[str]) -> List[Paper]:
        """Fetch structured metadata and abstracts for given PMIDs in batches via HTTP POST."""
        if not pmids:
            return []

        seen = set()
        clean_pmids = []
        for p in pmids:
            p_str = str(p).strip()
            if p_str and p_str not in seen:
                seen.add(p_str)
                clean_pmids.append(p_str)

        if not clean_pmids:
            return []

        batch_size = 200
        papers: List[Paper] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(clean_pmids), batch_size):
                batch = clean_pmids[i : i + batch_size]
                summary_data = {
                    **self._default_params(),
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "json",
                    "version": "2.0",
                }
                fetch_data = {
                    **self._default_params(),
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "rettype": "xml",
                    "retmode": "xml",
                }

                sum_res, fetch_res = await asyncio.gather(
                    client.post(f"{self.base_url}/esummary.fcgi", data=summary_data),
                    client.post(f"{self.base_url}/efetch.fcgi", data=fetch_data),
                    return_exceptions=True,
                )

                if isinstance(sum_res, Exception):
                    raise sum_res
                sum_res.raise_for_status()
                sum_dict = sum_res.json().get("result", {})

                abstracts: Dict[str, str] = {}
                if isinstance(fetch_res, httpx.Response) and fetch_res.status_code == 200:
                    abstracts = self._parse_pubmed_xml(fetch_res.text)

                for pmid in batch:
                    item = sum_dict.get(pmid)
                    if not item or not isinstance(item, dict):
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

                    abstract_text = abstracts.get(pmid, "") or title

                    pmc_val = item.get("pmcrefcount", 0)
                    pmc_count = int(pmc_val) if str(pmc_val).isdigit() else 0

                    paper = Paper(
                        id=pmid,
                        source="pubmed",
                        pmid=pmid,
                        title=title,
                        abstract=abstract_text,
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
        """Synchronously fetch structured metadata and abstracts in batches via HTTP POST."""
        if not pmids:
            return []

        seen = set()
        clean_pmids = []
        for p in pmids:
            p_str = str(p).strip()
            if p_str and p_str not in seen:
                seen.add(p_str)
                clean_pmids.append(p_str)

        if not clean_pmids:
            return []

        batch_size = 200
        papers: List[Paper] = []

        with httpx.Client(timeout=self.timeout) as client:
            for i in range(0, len(clean_pmids), batch_size):
                batch = clean_pmids[i : i + batch_size]
                summary_data = {
                    **self._default_params(),
                    "db": "pubmed",
                    "id": ",".join(batch),
                    "retmode": "json",
                    "version": "2.0",
                }
                sum_res = client.post(f"{self.base_url}/esummary.fcgi", data=summary_data)
                sum_res.raise_for_status()
                sum_dict = sum_res.json().get("result", {})

                abstracts: Dict[str, str] = {}
                try:
                    fetch_data = {
                        **self._default_params(),
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "rettype": "xml",
                        "retmode": "xml",
                    }
                    fetch_res = client.post(f"{self.base_url}/efetch.fcgi", data=fetch_data)
                    if fetch_res.status_code == 200:
                        abstracts = self._parse_pubmed_xml(fetch_res.text)
                except Exception:
                    pass

                for pmid in batch:
                    item = sum_dict.get(pmid)
                    if not item or not isinstance(item, dict):
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

                    abstract_text = abstracts.get(pmid, "") or title

                    pmc_val = item.get("pmcrefcount", 0)
                    pmc_count = int(pmc_val) if str(pmc_val).isdigit() else 0

                    paper = Paper(
                        id=pmid,
                        source="pubmed",
                        pmid=pmid,
                        title=title,
                        abstract=abstract_text,
                        authors=authors,
                        journal=journal,
                        pub_year=pub_year,
                        doi=doi,
                        is_open_access=bool(pmc_count > 0),
                        cited_by_count=pmc_count,
                    )
                    papers.append(paper)

        return papers
