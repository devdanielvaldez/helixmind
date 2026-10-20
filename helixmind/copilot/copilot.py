"""Interactive Research Copilot: bidirectional dialogue with active or completed investigations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from helixmind.agents.base import BaseAgent
from helixmind.config import get_settings
from helixmind.graph.models import KnowledgeGraph
from helixmind.journal.journal import ResearchJournal
from helixmind.llm.ollama_client import OllamaClient


class CopilotMessage(BaseModel):
    """A chat message in the investigation copilot dialogue."""
    sender: str = Field(description="'user' or 'copilot'")
    text: str
    timestamp: str


class CopilotResponse(BaseModel):
    """Response returned by the copilot including answers, citations, and triggered actions."""
    reply: str
    citations: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)


class ResearchCopilot:
    """Conversational intelligence for live or post-investigation steering."""

    # Active investigation conversation memory store
    _chat_histories: Dict[str, List[Dict[str, str]]] = {}

    def __init__(self, ollama_client: Optional[OllamaClient] = None, model: Optional[str] = None):
        self.settings = get_settings()
        self.client = ollama_client or OllamaClient()
        self.model = model or self.settings.ollama_default_model

    def get_conversation_history(self, investigation_id: str) -> List[Dict[str, str]]:
        if investigation_id not in self._chat_histories:
            self._chat_histories[investigation_id] = []
        return self._chat_histories[investigation_id]

    async def chat_async(
        self,
        investigation_id: str,
        user_message: str,
        topic: str = "",
        journal: Optional[ResearchJournal] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ) -> CopilotResponse:
        """Process a user query or steering instruction over an investigation."""
        history = self.get_conversation_history(investigation_id)

        # Context assembly from investigation state
        context_parts = []
        if topic:
            context_parts.append(f"Topic: {topic}")

        if journal and journal.evidence_dossier:
            context_parts.append(
                "Primary Evidence Dossier:\n"
                + "\n".join([f"- [{ev.paper_id}] {ev.paper_title}: {ev.key_excerpt[:150]}" for ev in journal.evidence_dossier[:4]])
            )

        if journal and journal.glossary:
            context_parts.append(
                "Key Discovered Terms: " + ", ".join([t.term for t in list(journal.glossary.values())[:8]])
            )

        if knowledge_graph and knowledge_graph.edges:
            context_parts.append(
                "Recent Knowledge Graph Connections:\n"
                + "\n".join([f"{e.source_id} -{e.relation}-> {e.target_id}" for e in knowledge_graph.edges[:5]])
            )

        system_prompt = (
            "You are Dr. Copilot, the Interactive Co-Investigator for HelixMind.\n"
            "You are conversing with the user while the investigation is running or after it has completed.\n"
            "Your goals:\n"
            "1. Answer scientific questions by citing exact papers, PMIDs, or evidence from the research dossier.\n"
            "2. Accept user steering commands (e.g. 'Investigate this side effect', 'Design primers for CD274', 'Challenge hypothesis 1').\n"
            "3. Provide actionable next steps and maintain a collegiate, sharp, academic tone.\n"
            "Keep responses concise, rigorous, and directly helpful."
        )

        agent = BaseAgent(
            name="Dr. Copilot",
            role="Interactive Research Co-Investigator",
            goal="Provide real-time steering and dialogue over the investigation state",
            system_prompt=system_prompt,
            model=self.model,
            temperature=0.3,
            ollama_client=self.client,
        )

        # Preload history into agent memory
        agent.memory = list(history[-8:])

        reply = await agent.execute_task_async(
            task_instruction=user_message,
            context="\n\n".join(context_parts),
        )

        # Record in history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})

        # Detect any citations mentioned
        import re
        citations = re.findall(r"\b(?:PMID:?\s*|NCT\d{8}|\b\d{8}\b)", reply)

        # Detect actions triggered
        actions = []
        lower_msg = user_message.lower()
        if "primer" in lower_msg or "crispr" in lower_msg:
            actions.append("Initiated in-silico CRISPR & PCR primer synthesis protocol")
        if "critique" in lower_msg or "riesgo" in lower_msg or "duda" in lower_msg:
            actions.append("Registered adversarial caveat in Knowledge Graph")
        if "profundiza" in lower_msg or "investiga" in lower_msg:
            actions.append("Dispatched targeted sub-inquiry to Research Director")

        return CopilotResponse(
            reply=reply,
            citations=list(set(citations)),
            actions_taken=actions,
        )
