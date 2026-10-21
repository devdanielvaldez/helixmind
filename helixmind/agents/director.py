"""Autonomous Research Director (Meta-Agent): determines knowledge gaps and spawns specialized agents on the fly."""

from __future__ import annotations

import re
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from helixmind.agents.base import BaseAgent
from helixmind.config import get_settings
from helixmind.graph.models import KnowledgeGraph
from helixmind.journal.journal import ResearchJournal
from helixmind.llm.ollama_client import OllamaClient
from helixmind.papers.europe_pmc_client import EuropePMCClient
from helixmind.papers.indexer import SemanticIndex
from helixmind.papers.models import Paper
from helixmind.papers.ncbi_client import NCBIClient


class AgentMission(BaseModel):
    """Dynamically generated specification for a new specialized agent."""
    agent_name: str = Field(description="Descriptive title/name (e.g. Dr. Immunogenomics, Dr. Lexicon, Dr. ResistanceHunter)")
    agent_role: str = Field(description="Scientific discipline or specialized functional role")
    agent_goal: str = Field(description="Precise scientific mission or question to resolve")
    task_type: str = Field(
        default="literature_deep_dive",
        description="literature_deep_dive, terminology_definition, pathway_mapping, clinical_gap_analysis, or adversarial_challenge",
    )
    sub_query_to_search: Optional[str] = Field(
        default=None,
        description="Targeted biomedical query to fetch fresh publications if more evidence is required",
    )
    system_prompt: str = Field(description="Tailored persona and domain expertise instructions for this agent")
    task_instruction: str = Field(description="Exact prompt to execute")
    rationale: str = Field(description="Scientific justification for why this agent is required at this stage")


class DirectorEvaluation(BaseModel):
    """The Research Director's strategic appraisal of current progress and next agent dispatch."""
    current_knowledge_assessment: str = Field(default="", description="Summary of verified findings so far")
    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="Key biological, clinical, or mechanistic unknowns remaining",
    )
    agent_name: str = Field(
        default="Dr. Lexicon",
        description="Creative name of next agent to spawn (e.g. Dr. BloodBrainBarrier, Dr. Microenvironment, Dr. Lexicon)",
    )
    agent_role: str = Field(default="Biomedical Specialist", description="Specialized domain role of the agent")
    agent_goal: str = Field(default="Investigate mechanistic pathways", description="Specific scientific objective")
    task_type: str = Field(
        default="literature_deep_dive",
        description="literature_deep_dive, terminology_definition, pathway_mapping, or safety_toxicology",
    )
    sub_query_to_search: Optional[str] = Field(
        default=None,
        description="Targeted biomedical query to fetch fresh publications if more evidence is required",
    )
    rationale: str = Field(default="Deepen mechanistic understanding", description="Why this agent is needed")
    investigation_ready_for_synthesis: bool = Field(
        default=False,
        description="True if evidence is comprehensive enough to author the final publication",
    )

    @field_validator("unresolved_questions", mode="before")
    @classmethod
    def clean_questions(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            parts = re.split(r"•|\n|-|;", v)
            return [p.strip().strip("'\"") for p in parts if p.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    def to_mission(self, topic: str) -> AgentMission:
        """Construct the executable AgentMission."""
        prompt = f"Investigate '{self.agent_goal}' in the context of '{topic}'. Synthesize 2 to 3 critical molecular findings."
        sys_prompt = f"You are {self.agent_name}, an expert {self.agent_role}. Focus deeply on {self.agent_goal}."
        return AgentMission(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            agent_goal=self.agent_goal,
            task_type=self.task_type,
            sub_query_to_search=self.sub_query_to_search,
            system_prompt=sys_prompt,
            task_instruction=prompt,
            rationale=self.rationale,
        )


class ResearchDirector:
    """Executive Meta-Agent coordinating progressive, deep multi-agent investigations."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None, model: Optional[str] = None):
        self.settings = get_settings()
        self.client = ollama_client or OllamaClient()
        self.model = model or self.settings.ollama_default_model

    def create_director_agent(self) -> BaseAgent:
        """Instantiate the executive director persona."""
        system_prompt = (
            "You are the Director of Autonomous Research, an elite Principal Investigator.\n"
            "Your role is to orchestrate deep scientific investigations by critically examining\n"
            "current findings, discovering gaps in molecular or clinical evidence, and deciding\n"
            "EXACTLY which specialized agent to spawn on the fly.\n"
            "Think freely: spawn ontologists, epigenomicists, tumor microenvironment specialists, "
            "pharmacologists, or bioinformaticians as needed."
        )
        return BaseAgent(
            name="Research Director",
            role="Principal Investigator & Executive Meta-Agent",
            goal="Identify critical knowledge gaps and dynamically spawn specialized agents to resolve them",
            system_prompt=system_prompt,
            model=self.model,
            temperature=0.3,
            ollama_client=self.client,
        )

    async def evaluate_and_dispatch_async(
        self,
        topic: str,
        cycle_number: int,
        max_cycles: int,
        journal: ResearchJournal,
        knowledge_graph: KnowledgeGraph,
        recent_findings_summary: str,
    ) -> DirectorEvaluation:
        """Analyze current research progress and dynamically specify the next agent to spawn."""
        director = self.create_director_agent()
        active_agents = [f"{a.name} ({a.role})" for a in journal.agents_roster]

        nodes_sample = [n.name for n in list(knowledge_graph.nodes.values())[:10]]
        edges_count = len(knowledge_graph.edges)

        director_prompt = (
            f"### Scientific Investigation Target: '{topic}'\n"
            f"### Investigation Depth Cycle: {cycle_number} of {max_cycles}\n"
            f"### Active Agents Previously Deployed: {', '.join(active_agents)}\n"
            f"### Knowledge Graph State: {len(knowledge_graph.nodes)} nodes ({', '.join(nodes_sample)}), {edges_count} edges\n\n"
            f"### Recent Findings & Observations:\n{recent_findings_summary[:900]}\n\n"
            "Assess current research progress and specify the NEXT specialized agent to spawn.\n"
            "Determine what functional role is missing (e.g. Dr. Lexicon to formalize ontology, "
            "Dr. TME for tumor microenvironment, Dr. Pharmacokinetics for bioavailability, "
            "Dr. ResistanceHunter for evasion mechanisms).\n"
            "If more literature is required on a sub-problem, provide 'sub_query_to_search'.\n"
            "Indicate if the investigation is ready for publication synthesis."
        )

        try:
            decision: DirectorEvaluation = await director.execute_structured_task_async(
                task_instruction=director_prompt,
                schema=DirectorEvaluation,
                temperature=0.3,
            )
            return decision.to_mission(topic)
        except Exception as e:
            director.log(f"Director structured evaluation fallback: {e}", style="yellow")
            if cycle_number == 1:
                return AgentMission(
                    agent_name="Dr. Lexicon",
                    agent_role="Biomedical Ontologist & Lexicographer",
                    agent_goal="Formalize biological definitions and acronyms for key discovered targets",
                    task_type="terminology_definition",
                    system_prompt="You are Dr. Lexicon, an expert in biomedical nomenclature and ontologies. Provide rigorous mechanistic definitions.",
                    task_instruction=f"Define the key molecular targets, signaling cascades, and acronyms identified in '{topic}'.",
                    rationale="Essential for grounding subsequent hypothesis generation in precise ontology.",
                )
            elif cycle_number == 2:
                return AgentMission(
                    agent_name="Dr. ResistanceHunter",
                    agent_role="Oncological Resistance Specialist",
                    agent_goal="Investigate microenvironmental and genetic resistance pathways",
                    task_type="literature_deep_dive",
                    sub_query_to_search=f"{topic} resistance mechanisms pathway",
                    system_prompt="You are Dr. ResistanceHunter, a specialist in therapeutic evasion and cellular resistance. Uncover escape pathways.",
                    task_instruction=f"Analyze how tumor cells develop resistance to therapy in '{topic}'.",
                    rationale="Dissecting resistance mechanisms is necessary for formulating dual-target synergistic hypotheses.",
                )
            else:
                return AgentMission(
                    agent_name="Dr. Pharmacovigilance",
                    agent_role="Translational Pharmacologist",
                    agent_goal="Analyze bioavailability, delivery platforms, and systemic toxicity",
                    task_type="safety_toxicology",
                    system_prompt="You are Dr. Pharmacovigilance, an expert pharmacologist. Assess off-target adverse risks.",
                    task_instruction=f"Evaluate drug delivery barriers, off-target binding, and toxicity profiles for targets in '{topic}'.",
                    rationale="Ensures formulated hypotheses are clinically viable and safe.",
                )

    async def execute_agent_mission_async(
        self,
        mission: AgentMission,
        index: SemanticIndex,
        journal: ResearchJournal,
        knowledge_graph: KnowledgeGraph,
        ncbi_client: NCBIClient,
        epmc_client: EuropePMCClient,
        active_papers: List[Paper],
    ) -> str:
        """Spawn the dynamic agent, execute targeted sub-searches if needed, and synthesize findings."""
        agent = BaseAgent(
            name=mission.agent_name,
            role=mission.agent_role,
            goal=mission.agent_goal,
            system_prompt=mission.system_prompt,
            model=self.model,
            temperature=0.3,
            ollama_client=self.client,
        )
        journal.record_agent_spawn(agent.name, agent.role, agent.goal, agent.model)
        agent.log(f"Mission Activated: '{mission.agent_goal}'", style="bold magenta")

        # 1. Targeted Sub-Search if requested by Director
        extra_context = ""
        if mission.sub_query_to_search:
            agent.log(f"Executing targeted sub-query: '{mission.sub_query_to_search}'...", style="cyan")
            try:
                sub_pmids = await ncbi_client.search_pubmed_async(mission.sub_query_to_search, limit=2)
                sub_papers = await ncbi_client.fetch_papers_async(sub_pmids)
                if sub_papers:
                    for sp in sub_papers:
                        if not any(p.id == sp.id for p in active_papers):
                            active_papers.append(sp)
                        agent.log(f"Reading retrieved publication: '{sp.title}' ({sp.pub_year or 'N/A'})", style="dim green")
                    await index.index_documents_async(sub_papers)
                    for sp in sub_papers:
                        journal.record_extracted_evidence(
                            paper_id=sp.pmid or sp.id,
                            paper_title=sp.title,
                            selection_rationale=f"Dynamically retrieved by {agent.name} for sub-mission: '{mission.agent_goal}'.",
                            key_excerpt=sp.abstract[:300].strip() if len(sp.abstract) > 30 else sp.title,
                            scientific_argumentation=f"Directly resolves knowledge gap identified by the Research Director concerning {mission.agent_role}.",
                            agent_name=agent.name,
                        )
                    extra_context = "\n".join([f"[{sp.id}] {sp.title}: {sp.abstract[:300]}" for sp in sub_papers])
                    journal.log(agent.name, "Targeted Sub-Search", f"Ingested {len(sub_papers)} papers on '{mission.sub_query_to_search}'")
            except Exception as ex:
                agent.log(f"Sub-query warning: {ex}", style="yellow")

        # 2. Retrieve targeted RAG evidence
        rag_hits = await index.get_relevant_context_async(mission.agent_goal, top_k=3)
        combined_context = f"{rag_hits}\n\n{extra_context}"

        # 3. Execute Mission Task
        finding = await agent.execute_task_async(
            task_instruction=mission.task_instruction,
            context=combined_context[:800],
        )

        agent.log(f"Findings & Discoveries:\n{finding.strip()}", style="bold green")
        journal.log(agent.name, mission.agent_role, finding[:250], category="discovery")

        # 4. If Terminology/Ontology Agent, parse defined terms into journal glossary
        if mission.task_type == "terminology_definition":
            for line in finding.split("\n"):
                if ":" in line or "-" in line:
                    parts = re.split(r"[:\-]", line, maxsplit=1)
                    if len(parts) == 2 and len(parts[0].strip()) > 1 and len(parts[1].strip()) > 5:
                        term_candidate = parts[0].strip().strip("*•_#")
                        def_candidate = parts[1].strip().strip("*_")
                        if len(term_candidate) < 30 and len(def_candidate) > 8:
                            journal.add_glossary_term(term_candidate, def_candidate, "target", agent.name)
                            knowledge_graph.add_node(term_candidate, "biomarker", description=def_candidate)

        return finding
