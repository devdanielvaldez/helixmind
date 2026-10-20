"""Continuous Research Journal and Ontological Glossary.

Records the full historical audit trail of agent activities, discoveries,
and terminology defined during an investigation.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class JournalEntry(BaseModel):
    """An event or discovery logged during research."""
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    agent_name: str
    stage: str
    message: str
    evidence_id: Optional[str] = None
    category: str = "observation"  # observation, discovery, critique, hypothesis, anomaly


class GlossaryTerm(BaseModel):
    """A biological, chemical, or medical term defined during the inquiry."""
    term: str
    definition: str
    category: str = "biological_target"  # biological_target, pathway, drug, phenotype, assay
    first_identified_by: str = "HelixMind"


class AgentRecord(BaseModel):
    """Registry entry for an agent spawned during the research."""
    name: str
    role: str
    goal: str
    model: str
    spawned_at: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class ResearchJournal(BaseModel):
    """Lab notebook tracking discoveries, active agents, and compiled glossary."""
    investigation_id: str
    topic: str
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    entries: List[JournalEntry] = Field(default_factory=list)
    glossary: Dict[str, GlossaryTerm] = Field(default_factory=dict)
    agents_roster: List[AgentRecord] = Field(default_factory=list)

    def log(
        self,
        agent_name: str,
        stage: str,
        message: str,
        evidence_id: Optional[str] = None,
        category: str = "observation",
    ) -> JournalEntry:
        """Add a chronological journal entry."""
        entry = JournalEntry(
            agent_name=agent_name,
            stage=stage,
            message=message,
            evidence_id=evidence_id,
            category=category,
        )
        self.entries.append(entry)
        return entry

    def record_agent_spawn(self, name: str, role: str, goal: str, model: str) -> None:
        """Register a newly spawned autonomous agent."""
        # Check if already registered
        if not any(a.name == name for a in self.agents_roster):
            self.agents_roster.append(
                AgentRecord(name=name, role=role, goal=goal, model=model)
            )

    def add_glossary_term(
        self,
        term: str,
        definition: str,
        category: str = "biological_target",
        defined_by: str = "HelixMind",
    ) -> GlossaryTerm:
        """Define a scientific term or acronym."""
        clean_key = term.strip().lower()
        if clean_key not in self.glossary:
            term_obj = GlossaryTerm(
                term=term.strip(),
                definition=definition.strip(),
                category=category,
                first_identified_by=defined_by,
            )
            self.glossary[clean_key] = term_obj
            return term_obj
        return self.glossary[clean_key]

    def to_markdown(self) -> str:
        """Render the complete research notebook and glossary in GitHub Markdown."""
        lines = [
            f"# Research Journal: {self.topic}",
            f"**Session ID:** `{self.investigation_id}` | **Started:** {self.created_at}\n",
            "## 1. Autonomous Agents Deployed",
            "| Agent | Specialization / Role | Objective | LLM Model | Spawned |",
            "|---|---|---|---|---|",
        ]

        for a in self.agents_roster:
            lines.append(f"| **{a.name}** | {a.role} | {a.goal} | `{a.model}` | {a.spawned_at} |")

        lines.extend([
            "\n## 2. Defined Scientific Terminology & Concepts",
            "| Term / Acronym | Category | Mechanistic Definition | Defined By |",
            "|---|---|---|---|",
        ])

        if self.glossary:
            for item in sorted(self.glossary.values(), key=lambda x: x.term):
                lines.append(f"| **{item.term}** | `{item.category}` | {item.definition} | {item.first_identified_by} |")
        else:
            lines.append("| *None recorded* | - | - | - |")

        lines.extend([
            "\n## 3. Chronological Discovery & Investigation Log",
            "| Time | Agent | Stage | Finding / Activity | Evidence ID |",
            "|---|---|---|---|---|",
        ])

        for e in self.entries:
            badge = f"**[{e.category.upper()}]**" if e.category != "observation" else ""
            cite = f"`{e.evidence_id}`" if e.evidence_id else "-"
            lines.append(f"| {e.timestamp} | {e.agent_name} | {e.stage} | {badge} {e.message} | {cite} |")

        return "\n".join(lines)

    def export_json(self, filepath: Path) -> None:
        """Save journal to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
