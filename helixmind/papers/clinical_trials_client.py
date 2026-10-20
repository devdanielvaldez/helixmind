"""ClinicalTrials.gov API v2 client for trial search, status, and outcome analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx

from helixmind.config import Settings, get_settings
from helixmind.papers.models import ClinicalStudy


class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov REST API v2."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.clinical_trials_base_url.rstrip("/")
        self.timeout = 30.0
        self.headers = {
            "User-Agent": "HelixMind/0.1.0 (Autonomous Scientific Research Engine; mailto:helixmind@example.com)",
            "Accept": "application/json",
        }

    async def search_studies_async(
        self,
        condition: str,
        intervention: Optional[str] = None,
        recruiting_only: bool = False,
        with_results: bool = False,
        phase: Optional[str] = None,
        limit: int = 10,
    ) -> List[ClinicalStudy]:
        """Asynchronously search clinical trials."""
        params: Dict[str, Any] = {
            "query.cond": condition,
            "pageSize": str(limit),
            "format": "json",
            "countTotal": "true",
        }

        if intervention:
            params["query.intr"] = intervention
        if recruiting_only:
            params["filter.overallStatus"] = "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING"
        if with_results:
            params["aggFilters"] = "results:with"
        if phase:
            params["filter.advanced"] = f"AREA[Phase]{phase}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(f"{self.base_url}/studies", params=params, headers=self.headers)
            res.raise_for_status()
            data = res.json()
            raw_studies = data.get("studies", [])

            studies: List[ClinicalStudy] = []
            for item in raw_studies:
                proto = item.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                design = proto.get("designModule", {})
                sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
                conditions_mod = proto.get("conditionsModule", {})
                arms_mod = proto.get("armsInterventionsModule", {})
                desc_mod = proto.get("descriptionModule", {})

                nct_id = ident.get("nctId", "NCT_UNKNOWN")
                title = ident.get("briefTitle", "")
                overall_status = status_mod.get("overallStatus", "UNKNOWN")
                phases = design.get("phases", [])
                phase_str = ", ".join(phases) if phases else None
                lead_sponsor = sponsor_mod.get("leadSponsor", {}).get("name")
                conditions = conditions_mod.get("conditions", [])

                interventions = []
                for intr in arms_mod.get("interventions", []):
                    name = intr.get("name")
                    if name:
                        interventions.append(name)

                brief_summary = desc_mod.get("briefSummary", "")
                has_results = bool(item.get("hasResults", False))

                studies.append(
                    ClinicalStudy(
                        nct_id=nct_id,
                        title=title,
                        status=overall_status,
                        phase=phase_str,
                        conditions=conditions,
                        interventions=interventions,
                        lead_sponsor=lead_sponsor,
                        brief_summary=brief_summary,
                        has_results=has_results,
                    )
                )

            return studies

    def search_studies(
        self,
        condition: str,
        intervention: Optional[str] = None,
        recruiting_only: bool = False,
        with_results: bool = False,
        phase: Optional[str] = None,
        limit: int = 10,
    ) -> List[ClinicalStudy]:
        """Synchronously search clinical trials."""
        params: Dict[str, Any] = {
            "query.cond": condition,
            "pageSize": str(limit),
            "format": "json",
            "countTotal": "true",
        }

        if intervention:
            params["query.intr"] = intervention
        if recruiting_only:
            params["filter.overallStatus"] = "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING"
        if with_results:
            params["aggFilters"] = "results:with"
        if phase:
            params["filter.advanced"] = f"AREA[Phase]{phase}"

        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.base_url}/studies", params=params, headers=self.headers)
            res.raise_for_status()
            data = res.json()
            raw_studies = data.get("studies", [])

            studies: List[ClinicalStudy] = []
            for item in raw_studies:
                proto = item.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                design = proto.get("designModule", {})
                sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
                conditions_mod = proto.get("conditionsModule", {})
                arms_mod = proto.get("armsInterventionsModule", {})
                desc_mod = proto.get("descriptionModule", {})

                nct_id = ident.get("nctId", "NCT_UNKNOWN")
                title = ident.get("briefTitle", "")
                overall_status = status_mod.get("overallStatus", "UNKNOWN")
                phases = design.get("phases", [])
                phase_str = ", ".join(phases) if phases else None
                lead_sponsor = sponsor_mod.get("leadSponsor", {}).get("name")
                conditions = conditions_mod.get("conditions", [])

                interventions = []
                for intr in arms_mod.get("interventions", []):
                    name = intr.get("name")
                    if name:
                        interventions.append(name)

                brief_summary = desc_mod.get("briefSummary", "")
                has_results = bool(item.get("hasResults", False))

                studies.append(
                    ClinicalStudy(
                        nct_id=nct_id,
                        title=title,
                        status=overall_status,
                        phase=phase_str,
                        conditions=conditions,
                        interventions=interventions,
                        lead_sponsor=lead_sponsor,
                        brief_summary=brief_summary,
                        has_results=has_results,
                    )
                )

            return studies
