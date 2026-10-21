"""Europe PMC client for preprints, open-access literature, and biological annotations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx

from helixmind.config import Settings, get_settings
from helixmind.papers.models import BiologicalEntity, Paper


class EuropePMCClient:
    """Client for Europe PMC REST API and Annotations API."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.europe_pmc_base_url.rstrip("/")
        self.annotations_base = self.settings.europe_pmc_annotations_url.rstrip("/")
        self.timeout = 30.0

    async def search_async(
        self,
        query: str,
        limit: int = 10,
        preprints_only: bool = False,
        open_access_only: bool = False,
        sort: str = "CITED desc",
    ) -> List[Paper]:
        """Asynchronously search Europe PMC with rich metadata and full abstracts."""
        final_query = query
        if preprints_only:
            final_query += ' AND SRC:"PPR"'
        if open_access_only:
            final_query += " AND OPEN_ACCESS:y"

        papers: List[Paper] = []
        cursor = "*"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while len(papers) < limit and cursor:
                batch_size = min(max(1, limit - len(papers)), 1000)
                params: Dict[str, Any] = {
                    "query": final_query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": str(batch_size),
                    "cursorMark": cursor,
                }
                if sort:
                    params["sort"] = sort

                res = await client.get(f"{self.base_url}/search", params=params)
                res.raise_for_status()
                data = res.json()
                raw_results = data.get("resultList", {}).get("result", [])
                if not raw_results:
                    break

                for item in raw_results:
                    paper_id = item.get("id", "")
                    title = item.get("title", "").rstrip(".")
                    abstract = item.get("abstractText", "") or title
                    author_string = item.get("authorString", "")
                    authors = [a.strip() for a in author_string.split(",") if a.strip()]

                    pmid = item.get("pmid")
                    pmcid = item.get("pmcid")
                    source = item.get("source", "MED")

                    pub_year = None
                    pub_year_raw = item.get("pubYear")
                    if pub_year_raw and str(pub_year_raw).isdigit():
                        pub_year = int(pub_year_raw)

                    paper = Paper(
                        id=f"{source}:{paper_id}",
                        source="europe_pmc" if source != "PPR" else "preprint",
                        pmid=pmid,
                        pmcid=pmcid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=item.get("journalTitle"),
                        pub_year=pub_year,
                        doi=item.get("doi"),
                        is_open_access=item.get("isOpenAccess") == "Y",
                        cited_by_count=item.get("citedByCount", 0),
                    )
                    papers.append(paper)
                    if len(papers) >= limit:
                        break

                next_cursor = data.get("nextCursorMark")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

            return papers

    def search(
        self,
        query: str,
        limit: int = 10,
        preprints_only: bool = False,
        open_access_only: bool = False,
        sort: str = "CITED desc",
    ) -> List[Paper]:
        """Synchronously search Europe PMC."""
        final_query = query
        if preprints_only:
            final_query += ' AND SRC:"PPR"'
        if open_access_only:
            final_query += " AND OPEN_ACCESS:y"

        papers: List[Paper] = []
        cursor = "*"

        with httpx.Client(timeout=self.timeout) as client:
            while len(papers) < limit and cursor:
                batch_size = min(max(1, limit - len(papers)), 1000)
                params: Dict[str, Any] = {
                    "query": final_query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": str(batch_size),
                    "cursorMark": cursor,
                }
                if sort:
                    params["sort"] = sort

                res = client.get(f"{self.base_url}/search", params=params)
                res.raise_for_status()
                data = res.json()
                raw_results = data.get("resultList", {}).get("result", [])
                if not raw_results:
                    break

                for item in raw_results:
                    paper_id = item.get("id", "")
                    title = item.get("title", "").rstrip(".")
                    abstract = item.get("abstractText", "") or title
                    author_string = item.get("authorString", "")
                    authors = [a.strip() for a in author_string.split(",") if a.strip()]

                    pmid = item.get("pmid")
                    pmcid = item.get("pmcid")
                    source = item.get("source", "MED")

                    pub_year = None
                    pub_year_raw = item.get("pubYear")
                    if pub_year_raw and str(pub_year_raw).isdigit():
                        pub_year = int(pub_year_raw)

                    paper = Paper(
                        id=f"{source}:{paper_id}",
                        source="europe_pmc" if source != "PPR" else "preprint",
                        pmid=pmid,
                        pmcid=pmcid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        journal=item.get("journalTitle"),
                        pub_year=pub_year,
                        doi=item.get("doi"),
                        is_open_access=item.get("isOpenAccess") == "Y",
                        cited_by_count=item.get("citedByCount", 0),
                    )
                    papers.append(paper)
                    if len(papers) >= limit:
                        break

                next_cursor = data.get("nextCursorMark")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

            return papers

    async def fetch_annotations_async(self, pmid: str) -> List[BiologicalEntity]:
        """Extract NLP-mined biological entities (genes, diseases, chemicals) for an article."""
        params = {"articleIds": f"MED:{pmid}", "format": "JSON"}
        headers = {"User-Agent": "HelixMind/0.1.0"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(
                    f"{self.annotations_base}/annotationsByArticleIds",
                    params=params,
                    headers=headers,
                )
                if res.status_code != 200:
                    return []
                items = res.json()
                if not isinstance(items, list) or not items:
                    return []

                entities: List[BiologicalEntity] = []
                seen = set()
                for record in items:
                    for ann in record.get("annotations", []):
                        name = ann.get("exact", "").strip()
                        cat = ann.get("type", "UNKNOWN").upper()
                        if name and name.lower() not in seen and len(name) > 2:
                            seen.add(name.lower())
                            entities.append(BiologicalEntity(name=name, entity_type=cat))
                return entities[:25]
        except Exception:
            return []
