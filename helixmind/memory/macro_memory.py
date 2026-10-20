"""Persistent Macro-Memory and Cross-Investigation Knowledge Base."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

from helixmind.config import get_settings
from helixmind.graph.models import KnowledgeGraph


class CrossInvestigationInsight(BaseModel):
    """A transferable biological discovery linking two distinct disease investigations."""
    shared_entity_name: str
    entity_category: str
    source_investigation_id: str
    source_topic: str
    current_topic: str
    transferable_rationale: str


class MacroMemory(BaseModel):
    """Global persistent knowledge repository bridging past and present research sessions."""
    total_investigations_indexed: int = 0
    macro_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    investigation_topics: Dict[str, str] = Field(default_factory=dict)
    topic_entities_index: Dict[str, Set[str]] = Field(default_factory=dict)

    @classmethod
    def get_storage_path(cls) -> Path:
        settings = get_settings()
        path = settings.data_dir / "memory" / "macro_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls) -> MacroMemory:
        """Load persistent macro-memory from disk or initialize fresh."""
        path = cls.get_storage_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert topic_entities_index lists to sets
                    if "topic_entities_index" in data:
                        data["topic_entities_index"] = {
                            k: set(v) for k, v in data["topic_entities_index"].items()
                        }
                    return cls.model_validate(data)
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        """Persist macro-memory to disk."""
        path = self.get_storage_path()
        dump_data = self.model_dump(mode="json")
        # Ensure sets are serialized to lists
        dump_data["topic_entities_index"] = {
            k: list(v) for k, v in self.topic_entities_index.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2)

    def merge_investigation(
        self,
        investigation_id: str,
        topic: str,
        session_graph: KnowledgeGraph,
    ) -> List[CrossInvestigationInsight]:
        """Ingest completed investigation graph, update macro-network, and surface cross-disease insights."""
        current_entities = set()
        insights: List[CrossInvestigationInsight] = []

        # Merge nodes into global macro-graph
        for nid, node in session_graph.nodes.items():
            current_entities.add(node.name.lower())
            self.macro_graph.add_node(
                name=node.name,
                category=node.category,
                description=node.description,
                statement=node.statement,
                reference=f"{topic} ({investigation_id})",
            )

        # Merge edges
        for edge in session_graph.edges:
            self.macro_graph.link_by_id(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation=edge.relation,
                evidence=edge.evidence,
                citation=f"{investigation_id}: {edge.citation or ''}",
                weight=edge.weight,
            )

        # Detect cross-investigation intersections with previous topics
        for prev_topic, prev_entities in self.topic_entities_index.items():
            if prev_topic.lower() == topic.lower():
                continue

            shared = current_entities.intersection(prev_entities)
            for shared_ent in shared:
                # Find matching node
                matching_node = next(
                    (n for n in self.macro_graph.nodes.values() if n.name.lower() == shared_ent),
                    None,
                )
                if matching_node and matching_node.category in ["protein", "gene", "pathway", "drug", "concept"]:
                    insights.append(
                        CrossInvestigationInsight(
                            shared_entity_name=matching_node.name,
                            entity_category=matching_node.category,
                            source_investigation_id=self.investigation_topics.get(prev_topic, "PREV-INV"),
                            source_topic=prev_topic,
                            current_topic=topic,
                            transferable_rationale=(
                                f"Shared molecular node '{matching_node.name}' ({matching_node.category}) "
                                f"was actively involved in '{prev_topic}'. Cross-disease mechanistic transfer "
                                f"suggests evaluating its therapeutic efficacy in '{topic}'."
                            ),
                        )
                    )

        # Update registry
        self.total_investigations_indexed += 1
        self.investigation_topics[topic] = investigation_id
        self.topic_entities_index[topic] = current_entities
        self.save()

        return insights[:5]
