"""Knowledge graph extraction from scientific abstracts and trial protocols."""

from __future__ import annotations

import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from helixmind.agents.base import BaseAgent
from helixmind.graph.models import KnowledgeGraph
from helixmind.llm.ollama_client import OllamaClient
from helixmind.papers.models import Paper


class ExtractedRelation(BaseModel):
    """An interaction extracted from biomedical text."""
    source_name: str = Field(default="", description="Source entity name (e.g. PD-L1, Tamoxifen, PI3K)")
    source_category: str = Field(default="protein", description="gene, protein, drug, pathway, disease, biomarker")
    relation: str = Field(default="interacts_with", description="inhibits, activates, upregulates, downregulates, targets, biomarker_of")
    target_name: str = Field(default="", description="Target entity name (e.g. T-cells, HER2, Apoptosis)")
    target_category: str = Field(default="process", description="gene, protein, pathway, disease, process")
    evidence: str = Field(default="", description="1 sentence mechanistic explanation")

    @model_validator(mode="before")
    @classmethod
    def clean_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            src = data.get("source_name") or data.get("source") or data.get("name") or data.get("from") or ""
            tgt = data.get("target_name") or data.get("target") or data.get("to") or ""
            rel = data.get("relation") or data.get("interaction") or data.get("type") or "interacts_with"
            data["source_name"] = str(src)
            data["target_name"] = str(tgt)
            data["relation"] = str(rel)
        elif isinstance(data, str):
            data = {"source_name": data, "target_name": "Target", "relation": "interacts_with"}
        return data

    @field_validator("source_category", "target_category", mode="before")
    @classmethod
    def clean_category(cls, v: Any) -> str:
        s = str(v).lower()
        valid = ["gene", "protein", "drug", "pathway", "disease", "biomarker", "process"]
        for cat in valid:
            if cat in s:
                return cat
        return "protein"


class ExtractedRelationList(BaseModel):
    """Container for batch extracted relations."""
    relations: List[ExtractedRelation] = Field(default_factory=list)

    @field_validator("relations", mode="before")
    @classmethod
    def clean_relations(cls, v: Any) -> Any:
        import json
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                v = parsed if isinstance(parsed, list) else [v]
            except Exception:
                v = [v]
        if isinstance(v, list):
            cleaned = []
            for item in v:
                if isinstance(item, dict):
                    cleaned.append(item)
                elif isinstance(item, str) and item.strip():
                    cleaned.append({"source_name": item.strip(), "target_name": "Target", "relation": "interacts_with"})
                elif isinstance(item, (int, float)):
                    cleaned.append({"source_name": f"Entity_{item}", "target_name": "Target", "relation": "interacts_with"})
            return cleaned
        return []


class GraphExtractor:
    """Extracts biological entities and relationships to populate the KnowledgeGraph."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None, model: Optional[str] = None):
        self.client = ollama_client or OllamaClient()
        self.model = model

    def create_graph_agent(self) -> BaseAgent:
        """Spawn Dr. Graph to extract mechanistic relationships from literature."""
        system_prompt = (
            "You are Dr. Graph, a Bioinformatician specializing in Knowledge Graphs and Systems Biology.\n"
            "Your job is to read scientific abstracts and extract causal biological relationships:\n"
            "(Entity A) -> [relation] -> (Entity B).\n"
            "Identify genes, proteins, pathways, drugs, and phenotypes."
        )
        return BaseAgent(
            name="Dr. Graph",
            role="Systems Biology & Knowledge Graph Specialist",
            goal="Extract validated biological networks, drug targets, and signaling cascades",
            system_prompt=system_prompt,
            model=self.model,
            temperature=0.2,
            ollama_client=self.client,
        )

    async def extract_from_papers_async(
        self,
        papers: List[Paper],
        graph: KnowledgeGraph,
        max_papers: int = 4,
    ) -> int:
        """Extract relationships from papers and add them to the KnowledgeGraph."""
        agent = self.create_graph_agent()
        total_added = 0

        # Also populate nodes from pre-mined Europe PMC biological entities and link co-occurrences
        for p in papers:
            for entity in p.entities:
                graph.add_node(
                    name=entity.name,
                    category=entity.entity_type.lower(),
                    reference=p.pmid or p.id,
                )
            if len(p.entities) >= 2:
                graph.add_edge(
                    source_name=p.entities[0].name,
                    source_category=p.entities[0].entity_type.lower(),
                    target_name=p.entities[1].name,
                    target_category=p.entities[1].entity_type.lower(),
                    relation="targets" if p.entities[0].entity_type.upper() == "DRUG" else "interacts_with",
                    evidence=f"Co-identified in {p.title}",
                    citation=p.pmid or p.id,
                )
                total_added += 1

        # Extract system-level relations in a single unified prompt
        valid_papers = [p for p in papers[:3] if p.abstract and len(p.abstract) > 60]
        if valid_papers:
            combined_text = "\n\n".join([f"[{p.id}] {p.title}: {p.abstract[:250]}" for p in valid_papers])
            prompt = (
                f"From these scientific findings:\n{combined_text}\n\n"
                "Extract 3 to 5 key biological interactions (e.g. Drug -> targets -> Protein, Gene -> activates -> Pathway).\n"
                "Return concrete entity names and specific relations."
            )
            try:
                result: ExtractedRelationList = await agent.execute_structured_task_async(
                    task_instruction=prompt,
                    schema=ExtractedRelationList,
                    temperature=0.2,
                )
                for rel in result.relations:
                    if rel.source_name and rel.target_name and rel.source_name.lower() != rel.target_name.lower():
                        graph.add_edge(
                            source_name=rel.source_name,
                            source_category=rel.source_category,
                            target_name=rel.target_name,
                            target_category=rel.target_category,
                            relation=rel.relation,
                            evidence=rel.evidence or "Systemic biological interaction",
                            citation=valid_papers[0].pmid or valid_papers[0].id,
                        )
                        total_added += 1
            except Exception:
                pass

        return total_added
