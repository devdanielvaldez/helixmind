"""Data models for scientific literature, clinical trials, and RAG document chunks."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class BiologicalEntity(BaseModel):
    """Mined or identified biological entity (gene, protein, drug, disease)."""
    name: str
    entity_type: str = Field(description="GENE, PROTEIN, CHEMICAL, DISEASE, or ORGANISM")
    ontology_id: Optional[str] = None


class Paper(BaseModel):
    """Standardized representation of a scientific publication or preprint."""
    id: str
    source: str = Field(default="pubmed", description="Originating database (MED, PMC, PPR)")
    title: str
    abstract: str
    authors: List[str] = Field(default_factory=list)
    journal: Optional[str] = None
    pub_year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    is_open_access: bool = False
    cited_by_count: int = 0
    full_text: Optional[str] = None
    mesh_terms: List[str] = Field(default_factory=list)
    entities: List[BiologicalEntity] = Field(default_factory=list)


class ClinicalStudy(BaseModel):
    """Standardized representation of an ongoing or completed clinical trial."""
    nct_id: str
    title: str
    status: str
    phase: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    lead_sponsor: Optional[str] = None
    brief_summary: str = ""
    has_results: bool = False
    results_summary: Optional[str] = None


class DocumentChunk(BaseModel):
    """A semantic text chunk indexed for vector RAG."""
    chunk_id: str
    doc_id: str
    doc_title: str
    source_type: Literal["paper", "trial", "annotation"]
    section: str = "abstract"
    content: str
    embedding: Optional[List[float]] = None
