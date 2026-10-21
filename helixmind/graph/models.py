"""Biomedical Knowledge Graph models and graph representations."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A biological or epistemic entity node in the knowledge graph."""
    id: str = Field(description="Unique normalized entity identifier (e.g. gene:her2, hypothesis:hyp_01)")
    name: str = Field(description="Display name of the entity")
    category: str = Field(
        default="protein",
        description="concept, evidence, hypothesis, theory, conclusion, doubt, question, gene, protein, pathway, drug, disease, biomarker, process",
    )
    description: Optional[str] = None
    statement: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """A directed semantic or epistemic relationship between two knowledge nodes."""
    source_id: str
    target_id: str
    relation: str = Field(
        description="supports, refutes, challenges, targets, explains, clarifies, derives_from, inhibits, activates, interacts_with, biomarker_of",
    )
    evidence: str = Field(default="", description="Snippet or mechanistic explanation")
    citation: Optional[str] = None
    weight: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def relation_type(self) -> str:
        """Alias for relation."""
        return self.relation

    @property
    def evidence_summary(self) -> str:
        """Alias for evidence."""
        return self.evidence


class KnowledgeGraph(BaseModel):
    """In-memory biomedical & epistemic knowledge graph connecting evidence, concepts, hypotheses, theories, and doubts."""
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
        statement: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> GraphNode:
        """Add or update an entity or epistemic node."""
        node_id = self.normalize_id(category, name)
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if reference and reference not in node.references:
                node.references.append(reference)
            if description and not node.description:
                node.description = description
            if statement and not node.statement:
                node.statement = statement
            return node

        refs = [reference] if reference else []
        node = GraphNode(
            id=node_id,
            name=name.strip(),
            category=category.lower(),
            description=description,
            statement=statement,
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
        return self.link_by_id(src_node.id, tgt_node.id, relation, evidence, citation, weight)

    def link_by_id(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        evidence: str = "",
        citation: Optional[str] = None,
        weight: float = 1.0,
    ) -> GraphEdge:
        """Link two existing node IDs with a directed relation."""
        clean_rel = relation.lower().replace(" ", "_")
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id and edge.relation == clean_rel:
                return edge

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=clean_rel,
            evidence=evidence,
            citation=citation,
            weight=weight,
        )
        self.edges.append(edge)
        return edge

    # -------------------------------------------------------------------------
    # Epistemic Knowledge Construction Helpers
    # -------------------------------------------------------------------------

    def add_epistemic_evidence(
        self,
        paper_id: str,
        paper_title: str,
        excerpt: str,
        rationale: str,
    ) -> GraphNode:
        """Register an empirical evidence node from an analyzed paper."""
        node_name = f"Evidence ({paper_id})"
        desc = f"Excerpt: {excerpt[:150]}... | Rationale: {rationale[:100]}"
        return self.add_node(
            name=node_name,
            category="evidence",
            description=desc,
            statement=excerpt,
            reference=paper_id,
        )

    def add_epistemic_hypothesis(
        self,
        hypo_id: str,
        title: str,
        mechanism: str,
        biological_targets: List[str],
        supporting_paper_ids: Optional[List[str]] = None,
    ) -> GraphNode:
        """Register a formulated hypothesis node and wire it to targets and supporting evidence."""
        hyp_node = self.add_node(
            name=f"{hypo_id}: {title[:40]}",
            category="hypothesis",
            description=mechanism[:200],
            statement=mechanism,
        )

        # Wire to biological targets
        for tgt in biological_targets:
            tgt_node = self.add_node(name=tgt, category="concept")
            self.link_by_id(
                source_id=hyp_node.id,
                target_id=tgt_node.id,
                relation="targets",
                evidence=f"Hypothesis {hypo_id} proposes intervention targeting {tgt}",
            )

        # Wire to supporting evidence nodes
        if supporting_paper_ids:
            for pid in supporting_paper_ids:
                ev_id = self.normalize_id("evidence", f"Evidence ({pid})")
                if ev_id in self.nodes:
                    self.link_by_id(
                        source_id=ev_id,
                        target_id=hyp_node.id,
                        relation="supports",
                        evidence=f"Publication {pid} provides empirical backing for {hypo_id}",
                    )

        return hyp_node

    def add_epistemic_doubt(
        self,
        doubt_statement: str,
        challenged_hypothesis_node_id: str,
        agent_name: str = "Dr. Skeptic",
    ) -> GraphNode:
        """Register a critical peer-review doubt/challenge node and connect it to the hypothesis."""
        slug_name = re.sub(r"[^a-zA-Z0-9]+", "_", doubt_statement[:30]).strip("_")
        doubt_node = self.add_node(
            name=f"Caveat: {doubt_statement[:45]}...",
            category="doubt",
            description=f"Raised by {agent_name}",
            statement=doubt_statement,
        )

        self.link_by_id(
            source_id=doubt_node.id,
            target_id=challenged_hypothesis_node_id,
            relation="challenges",
            evidence=doubt_statement,
        )
        return doubt_node

    def add_epistemic_conclusion(
        self,
        conclusion_title: str,
        statement: str,
        derived_from_hypothesis_node_id: Optional[str] = None,
    ) -> GraphNode:
        """Register an empirical conclusion or experimental milestone node."""
        conc_node = self.add_node(
            name=conclusion_title,
            category="conclusion",
            description=statement[:150],
            statement=statement,
        )

        if derived_from_hypothesis_node_id:
            self.link_by_id(
                source_id=conc_node.id,
                target_id=derived_from_hypothesis_node_id,
                relation="derives_from",
                evidence=statement,
            )
        return conc_node

    # -------------------------------------------------------------------------
    # Visual and Markdown Representations
    # -------------------------------------------------------------------------

    def to_mermaid(self, max_edges: int = 35) -> str:
        """Generate a valid Mermaid.js flowchart with distinct shapes and color styles for each epistemic type."""
        if not self.edges:
            return "```mermaid\ngraph LR\n    A[No Relations Discovered]\n```"

        lines = ["```mermaid", "flowchart TD"]

        selected_edges = self.edges[:max_edges]
        used_node_ids = set()
        for e in selected_edges:
            used_node_ids.add(e.source_id)
            used_node_ids.add(e.target_id)

        # Declare nodes with distinct brackets and semantic categories
        for nid in used_node_ids:
            if nid in self.nodes:
                node = self.nodes[nid]
                safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", nid)
                safe_name = node.name.replace('"', "'")

                cat = node.category.lower()
                if cat == "hypothesis":
                    lines.append(f'    {safe_id}{{{{"HYPOTHESIS: {safe_name}"}}}}')
                elif cat in ["doubt", "question"]:
                    lines.append(f'    {safe_id}[/"DOUBT / RISK: {safe_name}"/]')
                elif cat in ["evidence"]:
                    lines.append(f'    {safe_id}[("EVIDENCE: {safe_name}")]')
                elif cat in ["conclusion", "theory"]:
                    lines.append(f'    {safe_id}(["CONCLUSION: {safe_name}"])')
                else:
                    lines.append(f'    {safe_id}["{safe_name} ({node.category.upper()})"]')

        # Declare styled relations
        for e in selected_edges:
            src_safe = re.sub(r"[^a-zA-Z0-9_]", "_", e.source_id)
            tgt_safe = re.sub(r"[^a-zA-Z0-9_]", "_", e.target_id)
            rel_label = e.relation.replace("_", " ")

            if e.relation in ["challenges", "refutes"]:
                lines.append(f"    {src_safe} -.-|{rel_label}|-.-> {tgt_safe}")
            elif e.relation in ["supports", "validates"]:
                lines.append(f"    {src_safe} ==>|{rel_label}|==> {tgt_safe}")
            elif e.relation in ["derives_from"]:
                lines.append(f"    {src_safe} ==>|{rel_label}|==> {tgt_safe}")
            elif "inhibit" in e.relation or "downreg" in e.relation:
                lines.append(f"    {src_safe} --|{rel_label}|--> {tgt_safe}")
            else:
                lines.append(f"    {src_safe} -->|{rel_label}|--> {tgt_safe}")

        lines.append("```")
        return "\n".join(lines)

    def to_epistemic_matrix_markdown(self) -> str:
        """Format an epistemic knowledge matrix showing how evidence connects to hypotheses, doubts, and conclusions."""
        if not self.edges:
            return "*No epistemic connections mapped yet.*"

        rows = [
            "| Source Node | Category | Semantic Link | Target Node | Category | Mechanistic Evidence / Citation |",
            "|---|---|---|---|---|---|",
        ]
        for e in self.edges:
            src = self.nodes.get(e.source_id)
            tgt = self.nodes.get(e.target_id)
            if src and tgt:
                cite_str = e.citation or "Consortium Analysis"
                ev_snippet = (e.evidence[:70] + "...") if len(e.evidence) > 70 else (e.evidence or "Direct epistemic link")
                rows.append(
                    f"| **{src.name}** | `{src.category}` | **{e.relation.upper()}** | **{tgt.name}** | `{tgt.category}` | {ev_snippet} ({cite_str}) |"
                )
        return "\n".join(rows)

    def to_markdown_summary(self) -> str:
        """Backward-compatible markdown summary."""
        return self.to_epistemic_matrix_markdown()

    def export_json(self, filepath: Path) -> None:
        """Save graph to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
