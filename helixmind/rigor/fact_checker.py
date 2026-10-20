"""Anti-hallucination fact checker and evidence grounding auditor (Dr. Verifier)."""

from __future__ import annotations

import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from helixmind.agents.base import BaseAgent
from helixmind.journal.journal import ResearchJournal
from helixmind.llm.ollama_client import OllamaClient
from helixmind.papers.indexer import SemanticIndex


class AuditedClaim(BaseModel):
    """A claim audited for factual grounding against primary literature."""
    statement: str
    status: str = Field(description="VERIFIED_GROUNDED, PLAUSIBLE_INFERENCE, or SPECULATIVE_UNSUPPORTED")
    grounding_score: float = Field(ge=0.0, le=1.0)
    supporting_evidence: str = Field(default="")
    citation_id: Optional[str] = None


class FactCheckReport(BaseModel):
    """Comprehensive factual verification report for an investigation."""
    overall_rigor_score: float = Field(ge=0.0, le=1.0, description="Overall proportion of grounded factual assertions")
    total_claims_audited: int
    verified_claims_count: int
    speculative_claims_count: int
    claims: List[AuditedClaim] = Field(default_factory=list)
    verdict: str = Field(description="HIGHLY_RIGOROUS, ACCEPTABLE_WITH_INFERENCES, or HIGH_HALLUCINATION_RISK")


class FactChecker:
    """Audits scientific claims against RAG chunks and journal evidence to prevent hallucinations."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None, model: Optional[str] = None):
        self.client = ollama_client or OllamaClient()
        self.model = model

    def create_verifier_agent(self) -> BaseAgent:
        """Spawn Dr. Verifier to audit claims."""
        system_prompt = (
            "You are Dr. Verifier, Chief Scientific Auditor and Fact-Checker.\n"
            "Your sole objective is to detect scientific hallucinations and verify claims.\n"
            "Assess whether the hypothesis is grounded in empirical evidence, a plausible inference, or unsupported speculation."
        )
        return BaseAgent(
            name="Dr. Verifier",
            role="Chief Scientific Auditor & Fact-Checker",
            goal="Audit all claims and hypotheses against empirical literature to prevent hallucinations",
            system_prompt=system_prompt,
            model=self.model,
            temperature=0.1,
            ollama_client=self.client,
        )

    async def audit_hypothesis_async(
        self,
        hypothesis_title: str,
        mechanism_of_action: str,
        predictions: List[str],
        journal: ResearchJournal,
        index: SemanticIndex,
    ) -> FactCheckReport:
        """Verify empirical grounding of a hypothesis against indexed literature and evidence dossier."""
        claims_to_check = [
            hypothesis_title,
            mechanism_of_action[:180],
            *predictions[:2],
        ]

        # Assemble grounding reference from journal evidence dossier
        evidence_corpus = "\n".join(
            [f"[{ev.paper_id}] {ev.key_excerpt}" for ev in journal.evidence_dossier[:5]]
        )

        audited_claims: List[AuditedClaim] = []
        verified_count = 0

        for claim in claims_to_check:
            clean_claim = claim.strip()
            if len(clean_claim) < 15:
                continue

            # Query RAG for semantic matches
            rag_hits = await index.search_async(clean_claim, top_k=2)
            top_hit_score = rag_hits[0][1] if rag_hits else 0.0
            top_hit_doc = rag_hits[0][0].doc_id if rag_hits else None

            # Determine empirical grounding
            if top_hit_score >= 0.55 or any(word in evidence_corpus.lower() for word in clean_claim.lower().split() if len(word) > 5):
                status = "VERIFIED_GROUNDED"
                score = min(1.0, round(top_hit_score + 0.35, 2))
                verified_count += 1
                evidence_text = rag_hits[0][0].content[:120] if rag_hits else "Grounded in journal evidence dossier."
            elif top_hit_score >= 0.30:
                status = "PLAUSIBLE_INFERENCE"
                score = round(top_hit_score + 0.25, 2)
                verified_count += 1
                evidence_text = "Plausible mechanistic extrapolation from indexed literature."
            else:
                status = "SPECULATIVE_UNSUPPORTED"
                score = 0.35
                evidence_text = "Novel speculative deduction; requires targeted experimental validation."

            audited_claims.append(
                AuditedClaim(
                    statement=clean_claim,
                    status=status,
                    grounding_score=score,
                    supporting_evidence=evidence_text,
                    citation_id=top_hit_doc,
                )
            )

        total = len(audited_claims) or 1
        rigor = round(verified_count / total, 2)
        verdict = "HIGHLY_RIGOROUS" if rigor >= 0.8 else ("ACCEPTABLE_WITH_INFERENCES" if rigor >= 0.5 else "HIGH_HALLUCINATION_RISK")

        return FactCheckReport(
            overall_rigor_score=rigor,
            total_claims_audited=len(audited_claims),
            verified_claims_count=verified_count,
            speculative_claims_count=len(audited_claims) - verified_count,
            claims=audited_claims,
            verdict=verdict,
        )
