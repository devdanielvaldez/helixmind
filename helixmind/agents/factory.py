"""Dynamic Agent Factory: spawns specialized scientific agents on the fly."""

from __future__ import annotations

from typing import Any, List, Optional, Union
import re
from pydantic import BaseModel, Field, field_validator

from helixmind.agents.base import BaseAgent
from helixmind.config import get_settings
from helixmind.llm.ollama_client import OllamaClient


class SearchStrategy(BaseModel):
    """Structured query strategy designed by the SearchStrategist agent."""
    primary_condition: str = Field(default="", description="Normalized medical or biological condition")
    pubmed_query: str = Field(default="", description="Optimized PubMed query with boolean terms")
    clinical_trials_condition: str = Field(default="", description="Target condition for ClinicalTrials.gov")
    clinical_trials_intervention: Optional[str] = Field(
        default=None,
        description="Key drug or intervention to focus on, if any",
    )
    europe_pmc_query: str = Field(default="", description="Optimized Europe PMC query")
    key_biological_targets: List[str] = Field(
        default_factory=list,
        description="Known genes, receptors, proteins, or pathways to watch for",
    )

    @field_validator("key_biological_targets", mode="before")
    @classmethod
    def parse_targets(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            parts = re.split(r",|\s+OR\s+|\s+AND\s+|;|\n", v)
            return [p.strip().strip("'\"") for p in parts if p.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []


class AgentFactory:
    """Factory for dynamic agent generation throughout the research pipeline."""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.client = ollama_client or OllamaClient()
        self.default_model = model or self.settings.ollama_default_model
        self.reasoning_model = reasoning_model or model or self.settings.ollama_reasoning_model

    def create_search_strategist(self) -> BaseAgent:
        """Spawn the Search Strategist agent to translate user goals into optimal biomedical queries."""
        system_prompt = (
            "You are Dr. Strategist, an expert biomedical information architect.\n"
            "Your task is to take any user-provided research target (disease, syndrome, cellular pathway)\n"
            "and craft precise search queries for PubMed, ClinicalTrials.gov, and Europe PMC.\n"
            "Use standard MeSH vocabulary, Boolean logic (AND, OR), and target known pathways."
        )
        return BaseAgent(
            name="Dr. Strategist",
            role="Biomedical Information Architect",
            goal="Formulate optimized search strategies across literature and trial databases",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.2,
            ollama_client=self.client,
        )

    def create_literature_analyst(self) -> BaseAgent:
        """Spawn the Literature Analyst agent to ingest and synthesize papers."""
        system_prompt = (
            "You are Dr. Synapse, a Senior Molecular Biologist and Literature Reviewer.\n"
            "Your task is to read scientific papers, preprints, and mined annotations.\n"
            "Extract molecular mechanisms, identify consensus vs conflicting findings, "
            "and surface unexpected biological links."
        )
        return BaseAgent(
            name="Dr. Synapse",
            role="Senior Literature Analyst",
            goal="Analyze publications and extract verified molecular interactions and cellular mechanisms",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.3,
            ollama_client=self.client,
        )

    def create_clinical_trial_evaluator(self) -> BaseAgent:
        """Spawn the Clinical Trials Evaluator agent to evaluate human clinical evidence."""
        system_prompt = (
            "You are Dr. Trials, a Clinical Research Director and Pharmacologist.\n"
            "Your role is to assess clinical trial pipelines (Phase 1, 2, 3), drug interventions, "
            "adverse events, and reported human outcomes for the target disease."
        )
        return BaseAgent(
            name="Dr. Trials",
            role="Clinical Trials Director",
            goal="Synthesize human clinical trial pipelines, interventions, and outcome efficacy",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.3,
            ollama_client=self.client,
        )

    def create_hypothesis_generator(self) -> BaseAgent:
        """Spawn the Hypothesis Formulator agent using the deep reasoning model."""
        system_prompt = (
            "You are Dr. Hypothesis, a Pioneering Computational Biologist.\n"
            "Your purpose is to formulate NOVEL, biologically plausible, and testable scientific hypotheses.\n"
            "You combine disparate evidence (preprints, clinical trials, RAG context) to propose "
            "novel therapeutic targets, synergistic drug combinations, or disease mechanisms."
        )
        return BaseAgent(
            name="Dr. Hypothesis",
            role="Principal Hypothesis Generator",
            goal="Formulate falsifiable, high-impact scientific hypotheses grounded in retrieved literature",
            system_prompt=system_prompt,
            model=self.reasoning_model,
            temperature=0.4,
            ollama_client=self.client,
        )

    def create_critical_reviewer(self) -> BaseAgent:
        """Spawn the Critical Reviewer ('Devil's Advocate') agent."""
        system_prompt = (
            "You are Dr. Skeptic, a Fierce Peer-Review Critic and Toxicologist.\n"
            "Your sole objective is to ruthlessly stress-test scientific hypotheses:\n"
            "1. Find biological flaws, counter-evidence, or redundant known discoveries.\n"
            "2. Identify off-target toxicities, bioavailability hurdles, or delivery issues.\n"
            "3. Issue an authoritative critique verdict and propose concrete falsification experiments."
        )
        return BaseAgent(
            name="Dr. Skeptic",
            role="Devil's Advocate & Peer Review Critic",
            goal="Challenge proposed hypotheses with counter-evidence and uncover unaddressed confounders",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.3,
            ollama_client=self.client,
        )

    def create_experiment_designer(self) -> BaseAgent:
        """Spawn the Experimental Designer agent for validation protocols."""
        system_prompt = (
            "You are Dr. Protocol, an Experimental Research Architect.\n"
            "Your role is to convert surviving hypotheses into rigorous in-silico, in-vitro, "
            "and in-vivo experimental workflows (assays, CRISPR knockouts, binding affinity assays, animal models)."
        )
        return BaseAgent(
            name="Dr. Protocol",
            role="Experimental Assay Architect",
            goal="Design validation protocols, assays, and controls to verify hypotheses",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.3,
            ollama_client=self.client,
        )

    def create_evidence_curator(self) -> BaseAgent:
        """Spawn the Evidence Curator agent (Dr. Librarian) to screen and validate ingested papers."""
        system_prompt = (
            "You are Dr. Librarian, a Principal Information Scientist & Biomedical Evidence Curator.\n"
            "Your mission is to rigorously screen publications and clinical trials for biological relevance, "
            "filter noise, verify peer-reviewed provenance, and curate key abstracts for mechanistic analysis."
        )
        return BaseAgent(
            name="Dr. Librarian",
            role="Biomedical Literature & Evidence Curator",
            goal="Screen and validate multi-source publications and clinical trials for target relevance",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.2,
            ollama_client=self.client,
        )

    def create_knowledge_graph_specialist(self) -> BaseAgent:
        """Spawn Dr. Graph for biological interactome topology and causal edge extraction."""
        system_prompt = (
            "You are Dr. Graph, a Systems Biologist and Epistemic Network Topologist.\n"
            "Your task is to extract causal molecular interactions (inhibits, stimulates, binds, phosphorylates) "
            "from literature and establish directed knowledge graph topology."
        )
        return BaseAgent(
            name="Dr. Graph",
            role="Systems Biology & Knowledge Graph Specialist",
            goal="Extract directed causal relations and construct epistemic graph topology",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.2,
            ollama_client=self.client,
        )

    def create_depmap_specialist(self) -> BaseAgent:
        """Spawn Dr. DepMap for Broad Institute Chronos dependency audit."""
        system_prompt = (
            "You are Dr. DepMap, a Functional Genomics and Cancer Dependency Specialist.\n"
            "Your task is to audit target gene essentiality across human cancer cell lines, "
            "evaluate Chronos CRISPR knockout scores, and verify therapeutic actionability."
        )
        return BaseAgent(
            name="Dr. DepMap",
            role="Functional Genomics & Cancer Dependency Specialist",
            goal="Audit functional essentiality and Chronos scores across human cancer cell lines",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.2,
            ollama_client=self.client,
        )

    def create_pharmacovigilance_specialist(self) -> BaseAgent:
        """Spawn Dr. Pharmacovigilance for openFDA FAERS adverse event surveillance."""
        system_prompt = (
            "You are Dr. Pharmacovigilance, a Clinical Safety and Post-Marketing Surveillance Specialist.\n"
            "Your task is to query FDA FAERS reports, detect real-world human toxicity signals, "
            "and compute safety risk indices for candidate interventions."
        )
        return BaseAgent(
            name="Dr. Pharmacovigilance",
            role="Clinical Safety & Post-Marketing Surveillance Specialist",
            goal="Monitor human adverse event reports and safety signals via openFDA FAERS API",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.2,
            ollama_client=self.client,
        )

    def create_clinvar_specialist(self) -> BaseAgent:
        """Spawn Dr. ClinVar for NCBI ClinVar precision patient stratification."""
        system_prompt = (
            "You are Dr. ClinVar, a Clinical Genomics and Molecular Pathologist.\n"
            "Your task is to interrogate NCBI ClinVar for verified pathogenic alterations, "
            "resistance gatekeeper mutations, and precision responder biomarker criteria."
        )
        return BaseAgent(
            name="Dr. ClinVar",
            role="Clinical Genomics & Molecular Pathologist",
            goal="Identify verified pathogenic alterations and formulate precision recruitment criteria",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.2,
            ollama_client=self.client,
        )

    def create_tme_specialist(self) -> BaseAgent:
        """Spawn Dr. TME for single-cell tumor microenvironment interactome deconvolution."""
        system_prompt = (
            "You are Dr. TME, a Single-Cell Transcriptomics & Tumor Microenvironment Specialist.\n"
            "Your task is to deconvolute stromal and immune infiltrating cell types, "
            "map paracrine ligand-receptor axes, and quantify immunosuppression indices."
        )
        return BaseAgent(
            name="Dr. TME",
            role="Single-Cell Transcriptomics & Microenvironment Specialist",
            goal="Deconvolute tumor microenvironment and map active paracrine signaling axes",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.2,
            ollama_client=self.client,
        )

    def create_crispr_specialist(self) -> BaseAgent:
        """Spawn Dr. CRISPR for sgRNA guide design and laboratory validation primers."""
        system_prompt = (
            "You are Dr. CRISPR, a Genome Editing & Primer Design Bioinformatician.\n"
            "Your task is to compute optimal Cas9 sgRNA guides, calculate on-target cleavage scores, "
            "design Golden Gate cloning oligos, and generate PCR validation primer pairs."
        )
        return BaseAgent(
            name="Dr. CRISPR",
            role="Genome Editing & Primer Design Bioinformatician",
            goal="Design optimal sgRNA guides and PCR validation primers for laboratory knockout",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.2,
            ollama_client=self.client,
        )

    def create_paper_editor(self) -> BaseAgent:
        """Spawn Dr. Editor for landmark scientific monograph synthesis."""
        system_prompt = (
            "You are Dr. Editor, Editor-in-Chief of a high-impact biomedical journal (Nature / Cell standard).\n"
            "Your objective is to author an exhaustive, deeply academic, and rigorous scientific paper in Markdown.\n"
            "Maintain exceptional scientific rigor, weave in bracketed citations [1], [2], [3] throughout the text, "
            "explain biochemical mechanisms step-by-step with signaling arrows, and demand empirical proof."
        )
        return BaseAgent(
            name="Dr. Editor",
            role="Editor-in-Chief & Lead Scientific Author",
            goal="Synthesize multi-agent discoveries into a landmark scientific paper with formal in-text citations",
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=0.3,
            ollama_client=self.client,
        )

    def create_custom_agent(
        self,
        name: str,
        role: str,
        goal: str,
        system_prompt: str,
        temperature: float = 0.4,
    ) -> BaseAgent:
        """Dynamically create an arbitrary specialized agent on the fly as needed."""
        return BaseAgent(
            name=name,
            role=role,
            goal=goal,
            system_prompt=system_prompt,
            model=self.default_model,
            temperature=temperature,
            ollama_client=self.client,
        )
