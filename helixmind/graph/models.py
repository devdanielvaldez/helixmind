"""Biomedical Knowledge Graph models and graph representations."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A biological entity node in the knowledge graph."""
    id: str = Field(description="Unique normalized entity identifier (e.g. gene:her2)")
    name: str = Field(description="Display name of the entity")
    category: str = Field(
        default="protein",
        description="gene, protein, pathway, drug, disease, biomarker, or process",
    )
    description: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """A directed semantic relationship between two biological entities."""
    source_id: str
    target_id: str
    relation: str = Field(
        description="inhibits, activates, upregulates, downregulates, induces_resistance, targets, or associates_with",
    )
    evidence: str = Field(default="", description="Snippet or mechanistic explanation")
    citation: Optional[str] = None
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class KnowledgeGraph(BaseModel):
    """In-memory biomedical knowledge graph connecting entities, pathways, and evidence."""
    nodes: Dict[str, GraphNode] = Field(default_factory=dict)
    edges: List[GraphEdge] = Field(default_factory=list)

    @staticmethod
    def normalize_id(category: str, name: str) -> str:
        """Create a consistent slug identifier."""
        clean_name = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        clean_cat = category.strip().lower()
        return f"{clean_cat}:{clean_name}"

    def add_node(
        self,
        name: str,
        category: str = "protein",
        description: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> GraphNode:
        """Add or update an entity node."""
        node_id = self.normalize_id(category, name)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if reference and reference not in node.references:
                node.references.append(reference)
            if description and not node.description:
                node.description = description
            return node

        refs = [reference] if reference else []
        node = GraphNode(
            id=node_id,
            name=name.strip(),
            category=category.lower(),
            description=description,
            references=refs,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_name: str,
        source_category: str,
        target_name: str,
        target_category: str,
        relation: str,
        evidence: str = "",
        citation: Optional[str] = None,
        weight: float = 1.0,
    ) -> GraphEdge:
        """Add a directed relationship between two entities, creating nodes if needed."""
        src_node = self.add_node(source_name, source_category, reference=citation)
        tgt_node = self.add_node(target_name, target_category, reference=citation)

        # Check for existing duplicate edge
        for edge in self.edges:
            if edge.source_id == src_node.id and edge.target_id == tgt_node.id and edge.relation == relation:
                return edge

        edge = GraphEdge(
            source_id=src_node.id,
            target_id=tgt_node.id,
            relation=relation.lower(),
            evidence=evidence,
            citation=citation,
            weight=weight,
        )
        self.edges.append(edge)
        return edge

    def to_mermaid(self, max_edges: int = 25) -> str:
        """Generate a valid Mermaid.js flowchart for embedding directly into Markdown papers."""
        if not self.edges:
            return "```mermaid\ngraph LR\n    A[No Relations Discovered]\n```"

        lines = ["```mermaid", "flowchart TD"]

        # Category styling mappings
        styles = {
            "gene": "fill:#e1f5fe,stroke:#0288d1,stroke-width:2px",
            "protein": "fill:#e8f5e9,stroke:#388e3c,stroke-width:2px",
            "pathway": "fill:#fff3e0,stroke:#f57c00,stroke-width:2px",
            "drug": "fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px",
            "disease": "fill:#ffebee,stroke:#d32f2f,stroke-width:2px",
            "biomarker": "fill:#e0f2f1,stroke:#00796b,stroke-width:2px",
        }

        # Node declaration
        used_node_ids = set()
        selected_edges = self.edges[:max_edges]
        for e in selected_edges:
            used_node_ids.add(e.source_id)
            used_node_ids.add(e.target_id)

        for nid in used_node_ids:
            if nid in self.nodes:
                node = self.nodes[nid]
                safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", nid)
                safe_name = node.name.replace('"', "'")
                lines.append(f'    {safe_id}["{safe_name} ({node.category.upper()})"]')

        # Edges
        for e in selected_edges:
            src_safe = re.sub(r"[^a-zA-Z0-9_]", "_", e.source_id)
            tgt_safe = re.sub(r"[^a-zA-Z0-9_]", "_", e.target_id)
            rel_label = e.relation.replace("_", " ")

            if "inhibit" in e.relation or "downreg" in e.relation:
                lines.append(f"    {src_safe} --|{rel_label}|--> {tgt_safe}")
            else:
                lines.append(f"    {src_safe} == {rel_label} ==> {tgt_safe}")

        lines.append("```")
        return "\n".join(lines)

    def to_markdown_summary(self) -> str:
        """Format the graph as a Markdown table of verified biological interactions."""
        if not self.edges:
            return "No verified relational interactions discovered yet."

        rows = [
            "| Source Entity | Category | Relation | Target Entity | Category | Evidence / Citation |",
            "|---|---|---|---|---|---|",
        ]
        for e in self.edges:
            src = self.nodes.get(e.source_id)
            tgt = self.nodes.get(e.target_id)
            if src and tgt:
                cite_str = e.citation or "Literature"
                ev_str = (e.evidence[:60] + "...") if len(e.evidence) > 60 else (e.evidence or "Observed interaction")
                rows.append(
                    f"| **{src.name}** | `{src.category}` | *{e.relation}* | **{tgt.name}** | `{tgt.category}` | {ev_str} ({cite_str}) |"
                )
        return "\n".join(rows)

    def export_json(self, filepath: Path) -> None:
        """Save graph to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
