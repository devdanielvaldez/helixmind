"""Data models for scientific hypotheses, critiques, and final research reports."""

from __future__ import annotations

import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator


def _split_to_list(v: Any) -> List[str]:
    if isinstance(v, str):
        parts = re.split(r"•|\n|-|;|,", v)
        return [p.strip().strip("'\"") for p in parts if p.strip()]
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return []


class ScientificHypothesis(BaseModel):
    """A structured, falsifiable scientific hypothesis formulated by an agent."""
    id: str = Field(default="HYP-01", description="Unique hypothesis identifier (e.g. HYP-01)")
    title: str = Field(default="", description="Concise mechanistic title")
    target_condition: str = Field(default="", description="Target disease or phenotypic condition")
    biological_targets: List[str] = Field(
        default_factory=list,
        description="Receptors, genes, kinases, or signaling pathways involved",
    )
    mechanism_of_action: str = Field(
        default="",
        description="Step-by-step causal biological chain explaining the effect",
    )
    testable_predictions: List[str] = Field(
        default_factory=list,
        description="Concrete, empirical, falsifiable predictions",
    )
    validation_strategy: str = Field(
        default="",
        description="Suggested in-silico, in-vitro, or in-vivo assay/experiment",
    )
    confidence_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Estimated confidence score based on literature support",
    )
    supporting_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Citations, PMIDs, or NCT IDs supporting the hypothesis",
    )
    counter_arguments: List[str] = Field(
        default_factory=list,
        description="Potential contradictions, adverse reactions, or off-target effects",
    )
    critique_verdict: Optional[str] = Field(
        default=None,
        description="Verdict from Devil's Advocate review agent",
    )
    recommended_experiments: List[str] = Field(
        default_factory=list,
        description="Critical experiments to validate or falsify the hypothesis",
    )

    @field_validator("id", mode="before")
    @classmethod
    def clean_id(cls, v: Any) -> str:
        if isinstance(v, (int, float)):
            return f"HYP-{int(v):02d}"
        return str(v) if v else "HYP-01"

    @field_validator("confidence_score", mode="before")
    @classmethod
    def clean_score(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v) if v <= 1.0 else float(v) / 100.0
        if isinstance(v, str):
            clean = v.replace("%", "").strip()
            if clean.replace(".", "", 1).isdigit():
                val = float(clean)
                return val if val <= 1.0 else val / 100.0
        return 0.75

    @field_validator(
        "biological_targets",
        "testable_predictions",
        "supporting_evidence_ids",
        "counter_arguments",
        "recommended_experiments",
        mode="before",
    )
    @classmethod
    def clean_lists(cls, v: Any) -> List[str]:
        return _split_to_list(v)


class HypothesisList(BaseModel):
    """Container for batch hypothesis generation."""
    hypotheses: List[ScientificHypothesis] = Field(default_factory=list)


class HypothesisCritique(BaseModel):
    """Critical peer-review response for a scientific hypothesis."""
    hypothesis_id: str = Field(default="HYP-01")
    verdict: str = Field(
        default="Plausible with Caveats",
        description="Strongly Supported, Plausible with Caveats, Highly Speculative, or Likely Flawed",
    )
    biological_plausibility_score: float = Field(default=0.7, ge=0.0, le=1.0)
    identified_risks: List[str] = Field(default_factory=list)
    unaddressed_confounders: List[str] = Field(default_factory=list)
    falsification_tests: List[str] = Field(default_factory=list)

    @field_validator("identified_risks", "unaddressed_confounders", "falsification_tests", mode="before")
    @classmethod
    def clean_critique_lists(cls, v: Any) -> List[str]:
        return _split_to_list(v)


class ResearchReport(BaseModel):
    """Comprehensive autonomous research investigation report."""
    topic: str
    timestamp: str
    papers_analyzed: int
    clinical_trials_analyzed: int
    entities_discovered: List[str] = Field(default_factory=list)
    hypotheses: List[ScientificHypothesis] = Field(default_factory=list)
    clinical_landscape_summary: str = ""
    critical_synthesis: str = ""
    next_research_milestones: List[str] = Field(default_factory=list)
    paper_markdown_path: Optional[str] = None
    knowledge_graph_summary: Optional[str] = None
    depmap_evaluations: List[Dict[str, Any]] = Field(default_factory=list)
    denovo_analogs: List[Dict[str, Any]] = Field(default_factory=list)
    clinical_simulation: Optional[Dict[str, Any]] = None
