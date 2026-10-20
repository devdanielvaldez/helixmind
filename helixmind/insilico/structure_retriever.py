"""PDB and AlphaFold 3D protein structure retrieval engine."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import httpx
from pydantic import BaseModel, Field

from helixmind.config import get_settings

TARGET_PDB_MAPPINGS: Dict[str, str] = {
    "her2": "1N8Z",
    "erbb2": "1N8Z",
    "pd-l1": "5X8M",
    "pdl1": "5X8M",
    "cd274": "5X8M",
    "pd-1": "4ZQK",
    "pd1": "4ZQK",
    "egfr": "1M17",
    "ctla-4": "1I85",
    "ctla4": "1I85",
    "kras": "7E27",
    "stat3": "6TLC",
    "idh1": "4UMX",
    "braf": "4MNE",
    "brca1": "1JNX",
    "vegfr2": "4ASD",
    "pik3ca": "4JPS",
    "cdk4": "2W96",
    "cdk6": "1BLX",
    "trop2": "7X7S",
}


class ProteinStructure(BaseModel):
    """3D structure metadata and local filesystem storage reference."""
    target_name: str
    pdb_id: str
    source: str = "rcsb_pdb"
    title: str = ""
    resolution_angstroms: Optional[float] = None
    structure_file_path: Optional[str] = None
    view_url: str = ""


class StructureRetriever:
    """Retrieves high-resolution crystallographic structures from RCSB PDB."""

    def __init__(self, structures_dir: Optional[Path] = None):
        settings = get_settings()
        self.structures_dir = structures_dir or (settings.data_dir / "structures")
        self.structures_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = 20.0

    @staticmethod
    def resolve_pdb_id(target_name: str) -> Optional[str]:
        """Resolve any target protein name to its canonical PDB ID via RCSB Search API v2 or local cache."""
        clean = target_name.strip().lower()
        if clean in TARGET_PDB_MAPPINGS:
            return TARGET_PDB_MAPPINGS[clean]

        # Dynamic search on RCSB PDB v2 API
        try:
            query = {
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": target_name.strip()},
                },
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": 1}},
            }
            with httpx.Client(timeout=5.0) as client:
                res = client.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
                if res.status_code == 200:
                    results = res.json().get("result_set", [])
                    if results:
                        pdb_id = results[0].get("identifier")
                        if pdb_id:
                            TARGET_PDB_MAPPINGS[clean] = pdb_id
                            return pdb_id
        except Exception:
            pass

        for k, pdb_id in TARGET_PDB_MAPPINGS.items():
            if k in clean or clean in k:
                return pdb_id
        return None

    async def fetch_structure_async(self, target_name: str) -> Optional[ProteinStructure]:
        """Download PDB file and metadata for a given receptor or protein."""
        pdb_id = self.resolve_pdb_id(target_name)
        if not pdb_id:
            return None

        pdb_file = self.structures_dir / f"{pdb_id}.pdb"
        view_url = f"https://www.rcsb.org/structure/{pdb_id}"

        # If already cached locally, return existing record
        if pdb_file.exists() and pdb_file.stat().st_size > 1000:
            return ProteinStructure(
                target_name=target_name.upper(),
                pdb_id=pdb_id,
                source="rcsb_pdb",
                title=f"Crystallographic structure of {target_name.upper()}",
                structure_file_path=str(pdb_file),
                view_url=view_url,
            )

        # Download from RCSB PDB
        download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(download_url)
                if res.status_code == 200:
                    with open(pdb_file, "w", encoding="utf-8") as f:
                        f.write(res.text)
                    return ProteinStructure(
                        target_name=target_name.upper(),
                        pdb_id=pdb_id,
                        source="rcsb_pdb",
                        title=f"Crystallographic structure of {target_name.upper()}",
                        structure_file_path=str(pdb_file),
                        view_url=view_url,
                    )
        except Exception:
            pass

        return None
