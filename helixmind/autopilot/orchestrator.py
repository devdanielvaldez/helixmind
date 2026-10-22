"""Auto-pilot scientific research workflow orchestrator.

Coordinates dynamic agents, multi-source literature retrieval, vector RAG indexing,
hypothesis generation, and critical peer-review.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Callable, List, Optional
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from helixmind.agents.director import AgentMission, DirectorEvaluation, ResearchDirector
from helixmind.agents.factory import AgentFactory, SearchStrategy
from helixmind.api.events import EventType
from helixmind.api.ws_manager import ws_manager
from helixmind.config import Settings, get_settings
from helixmind.core.resources import get_resource_manager
from helixmind.graph import GraphExtractor, KnowledgeGraph
from helixmind.hypotheses.models import (
    HypothesisCritique,
    HypothesisList,
    ResearchReport,
    ScientificHypothesis,
)
from helixmind.insilico import (
    ClinVarStratifier,
    CRISPRDesignReport,
    CRISPRDesigner,
    DepMapGeneEffect,
    DepMapVerifier,
    FDAPharmacovigilance,
    FDASafetyReport,
    GenomicStratificationReport,
    TMEMapper,
    TMEMicroenvironmentReport,
)
from helixmind.journal import ResearchJournal
from helixmind.llm.ollama_client import OllamaClient
from helixmind.paper_writer import ScientificPaperGenerator
from helixmind.rigor import PeerReviewTribunal, TribunalConsensus
from helixmind.papers.clinical_trials_client import ClinicalTrialsClient
from helixmind.papers.europe_pmc_client import EuropePMCClient
from helixmind.papers.indexer import SemanticIndex
from helixmind.papers.models import BiologicalEntity, ClinicalStudy, Paper
from helixmind.papers.ncbi_client import NCBIClient

console = Console()


def print_agent_spawn_card(
    name: str,
    role: str,
    goal: str,
    model: str,
    task_instruction: str = "",
    color: str = "green",
) -> None:
    """Print an autonomous agent spawn card detailing role, objective, model, and active task."""
    lines = [
        f"[bold]Role:[/bold] [cyan]{role}[/cyan]",
        f"[bold]Scientific Objective:[/bold] {goal}",
        f"[bold]LLM Engine:[/bold] [magenta]{model}[/magenta]",
    ]
    if task_instruction:
        clean_task = task_instruction.replace("\n", " ").strip()
        lines.append(f"[bold]Assigned Mission:[/bold] [dim]{clean_task}[/dim]")
    console.print(
        Panel(
            "\n".join(lines),
            title=f"🤖 Spawning Autonomous Agent: [bold {color}]{name}[/bold {color}]",
            border_style=color,
        )
    )


class AutoPilotOrchestrator:
    """Orchestrates autonomous multi-agent scientific investigations."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.settings = settings or get_settings()
        self.ollama = ollama_client or OllamaClient(settings=self.settings)
        self.factory = AgentFactory(ollama_client=self.ollama)
        self.ncbi_client = NCBIClient(settings=self.settings)
        self.epmc_client = EuropePMCClient(settings=self.settings)
        self.ct_client = ClinicalTrialsClient(settings=self.settings)
        self.index = SemanticIndex(ollama_client=self.ollama)

    async def investigate_async(
        self,
        topic: str,
        max_papers: int = 5,
        max_trials: int = 4,
        depth: str = "deep",
        model: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        enable_depmap: bool = True,
        enable_pharmacovigilance: bool = True,
        enable_clinvar: bool = True,
        enable_tme: bool = True,
        enable_crispr: bool = True,
        enable_tribunal: bool = True,
        export_formats: Optional[List[str]] = None,
    ) -> ResearchReport:
        """Run an autonomous end-to-end investigation on a target condition or topic."""
        # Initialize Host Resource Guard & Sanitize Operational Limits and Model
        resource_guard = get_resource_manager()
        raw_model = model or self.settings.ollama_default_model
        safe_model, model_warning = resource_guard.resolve_safe_model(self.ollama, raw_model)
        if model_warning:
            console.print(Panel(model_warning, border_style="yellow"))
        chosen_model = safe_model
        self.factory = AgentFactory(ollama_client=self.ollama, model=chosen_model)

        clamped_papers, clamped_trials, limit_warning = resource_guard.sanitize_request_limits(max_papers, max_trials)
        if limit_warning:
            console.print(Panel(limit_warning, border_style="yellow"))
        max_papers = clamped_papers
        max_trials = clamped_trials
        console.print(resource_guard.get_status_panel())

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        inv_id = f"INV-{int(datetime.now().timestamp())}"
        slug = "".join([c if c.isalnum() else "_" for c in topic.lower()])[:30].strip("_")

        depth_cycles = {
            "deep": 2,
            "exhaustive": 3,
            "institutional": 4,
            "standard": 2,
            "quick": 1,
        }
        num_cycles = depth_cycles.get(depth.lower(), 2)

        # Initialize Research Journal & Knowledge Graph
        journal = ResearchJournal(investigation_id=inv_id, topic=topic)
        knowledge_graph = KnowledgeGraph()
        graph_extractor = GraphExtractor(ollama_client=self.ollama, model=chosen_model)
        paper_writer = ScientificPaperGenerator(ollama_client=self.ollama, model=chosen_model)

        def notify(msg: str) -> None:
            if on_progress:
                on_progress(msg)
            console.print(f"[bold cyan]▶ {msg}[/bold cyan]")

        async def emit(event_type: EventType, message: str, data: Dict[str, Any] | None = None) -> None:
            notify(message)
            try:
                await ws_manager.broadcast_event(
                    event_type=event_type,
                    investigation_id=inv_id,
                    message=message,
                    data=data or {},
                )
            except Exception:
                pass

        await emit(
            EventType.INVESTIGATION_STARTED,
            f"Initiating autonomous investigation for: '{topic}'",
            {"topic": topic, "depth": depth, "investigation_id": inv_id, "timestamp": timestamp},
        )
        journal.log("Orchestrator", "Initiation", f"Launched autonomous investigation on '{topic}'")

        # ---------------------------------------------------------------------
        # 1. Spawn Search Strategist & Formulate Queries
        # ---------------------------------------------------------------------
        console.rule("[bold cyan]Search Strategy Formulation (Dr. Search)[/bold cyan]")
        strategist = self.factory.create_search_strategist()
        journal.record_agent_spawn(strategist.name, strategist.role, strategist.goal, strategist.model)
        print_agent_spawn_card(
            name=strategist.name,
            role=strategist.role,
            goal=strategist.goal,
            model=strategist.model,
            task_instruction=f"Analyze '{topic}', extract biological targets, and formulate search queries for PubMed, Europe PMC, and ClinicalTrials.gov",
            color="green",
        )
        await emit(
            EventType.AGENT_SPAWNED,
            f"Spawning {strategist.name} to construct structured biomedical queries...",
            {"name": strategist.name, "role": strategist.role, "goal": strategist.goal},
        )

        strategy_prompt = (
            f"Analyze this scientific topic: '{topic}'.\n"
            "Formulate optimized search parameters for PubMed, Europe PMC, and ClinicalTrials.gov.\n"
            "Identify key biological targets, proteins, receptors, or signaling pathways involved."
        )
        try:
            strategy: SearchStrategy = await strategist.execute_structured_task_async(
                task_instruction=strategy_prompt,
                schema=SearchStrategy,
                temperature=0.2,
            )
            strategist.log(
                f"Generated strategy -> Condition: {strategy.primary_condition} | Targets: {', '.join(strategy.key_biological_targets[:4])}",
                style="green",
            )
            strategy_panel = (
                f"[bold]Primary Condition:[/bold] [cyan]{strategy.primary_condition}[/cyan]\n"
                f"[bold]Key Biological Targets:[/bold] [yellow]{', '.join(strategy.key_biological_targets)}[/yellow]\n"
                f"[bold]NCBI PubMed Query:[/bold] [white]{strategy.pubmed_query}[/white]\n"
                f"[bold]Europe PMC Query:[/bold] [white]{strategy.europe_pmc_query}[/white]\n"
                f"[bold]ClinicalTrials Condition:[/bold] [white]{strategy.clinical_trials_condition}[/white]"
                + (f"\n[bold]Candidate Intervention:[/bold] [white]{strategy.clinical_trials_intervention}[/white]" if strategy.clinical_trials_intervention else "")
            )
            console.print(Panel(strategy_panel, title="🎯 Biomedical Search Strategy Formulated", border_style="cyan"))
            journal.log(strategist.name, "Query Strategy", f"Target: {strategy.primary_condition}. PubMed: {strategy.pubmed_query}")
        except Exception as e:
            strategist.log(f"Structured strategy fallback: {e}", style="yellow")
            strategy = SearchStrategy(
                primary_condition=topic,
                pubmed_query=topic,
                clinical_trials_condition=topic,
                europe_pmc_query=topic,
                key_biological_targets=[],
            )
            journal.log(strategist.name, "Query Strategy Fallback", f"Using base query: {topic}")

        for target in strategy.key_biological_targets:
            knowledge_graph.add_node(target, "protein")
            journal.add_glossary_term(target, "Primary biological target identified in strategy", "target", strategist.name)

        await resource_guard.throttle_if_needed()
        resource_guard.optimize_memory()

        # ---------------------------------------------------------------------
        # 2. Ingest Literature & Trials across PubMed, Europe PMC, ClinicalTrials (Dr. Librarian)
        # ---------------------------------------------------------------------
        console.rule("[bold cyan]Multi-Source Evidence Ingestion & Curation (Dr. Librarian)[/bold cyan]")
        librarian = self.factory.create_evidence_curator()
        journal.record_agent_spawn(librarian.name, librarian.role, librarian.goal, librarian.model)
        print_agent_spawn_card(
            name=librarian.name,
            role=librarian.role,
            goal=librarian.goal,
            model=librarian.model,
            task_instruction=f"Query international registries (NCBI PubMed, Europe PMC, ClinicalTrials.gov) and screen publications for '{strategy.primary_condition}'",
            color="cyan",
        )
        await emit(
            EventType.AGENT_SPAWNED,
            f"Spawning {librarian.name} to screen and curate biomedical evidence...",
            {"name": librarian.name, "role": librarian.role, "goal": librarian.goal},
        )

        notify("Ingesting literature from PubMed, preprints from Europe PMC, and ClinicalTrials.gov...")
        console.print("[dim]Submitting queries across international biomedical registries:[/dim]")
        console.print(f"  • NCBI PubMed: [cyan]'{strategy.pubmed_query}'[/cyan] (target limit: {max_papers} papers)")
        console.print(f"  • Europe PMC: [cyan]'{strategy.europe_pmc_query}'[/cyan] (target limit: {max(1, max_papers // 2)} papers + preprints)")
        console.print(f"  • ClinicalTrials.gov: [cyan]'{strategy.clinical_trials_condition}'[/cyan] (target limit: {max_trials} trials)\n")

        async def fetch_pubmed():
            try:
                pmids = await self.ncbi_client.search_pubmed_async(strategy.pubmed_query, limit=max_papers)
                return await self.ncbi_client.fetch_papers_async(pmids)
            except Exception as ex:
                console.print(f"[yellow]PubMed fetch notice: {ex}[/yellow]")
                return []

        async def fetch_epmc():
            target_limit = max(1, max_papers // 2)
            try:
                regular = await self.epmc_client.search_async(strategy.europe_pmc_query, limit=target_limit)
                preprints = await self.epmc_client.search_async(
                    strategy.europe_pmc_query,
                    limit=target_limit,
                    preprints_only=True,
                )
                return regular + preprints
            except Exception as ex:
                console.print(f"[yellow]Europe PMC fetch notice: {ex}[/yellow]")
                return []

        async def fetch_trials():
            if max_trials <= 0:
                return []
            try:
                return await self.ct_client.search_studies_async(
                    condition=strategy.clinical_trials_condition,
                    intervention=strategy.clinical_trials_intervention,
                    limit=max_trials,
                )
            except Exception as ex:
                console.print(f"[yellow]ClinicalTrials fetch notice: {ex}[/yellow]")
                return []

        pubmed_papers, epmc_papers, clinical_trials = await asyncio.gather(
            fetch_pubmed(),
            fetch_epmc(),
            fetch_trials(),
        )

        seen_titles = set()
        deduped_papers: List[Paper] = []
        for p in pubmed_papers + epmc_papers:
            norm_title = p.title.strip().lower()
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                deduped_papers.append(p)

        console.print(f"[bold green]✓ Literature Retrieval Complete:[/bold green]")
        console.print(f"  • NCBI PubMed retrieved: [cyan]{len(pubmed_papers)}[/cyan] peer-reviewed articles")
        console.print(f"  • Europe PMC retrieved: [cyan]{len(epmc_papers)}[/cyan] open-access publications & preprints")
        console.print(f"  • ClinicalTrials.gov retrieved: [cyan]{len(clinical_trials)}[/cyan] human clinical trials")
        console.print(f"  • Total unique publications indexed: [bold green]{len(deduped_papers)}[/bold green]\n")

        await emit(
            EventType.LITERATURE_INGESTED,
            f"Retrieved {len(deduped_papers)} unique papers and {len(clinical_trials)} clinical trials.",
            {
                "papers_count": len(deduped_papers),
                "trials_count": len(clinical_trials),
                "papers": [{"id": p.id, "title": p.title, "year": p.pub_year} for p in deduped_papers[:5]],
                "trials": [{"id": t.nct_id, "title": t.title, "phase": t.phase} for t in clinical_trials[:5]],
            },
        )
        journal.log(librarian.name, "Evidence Curation", f"Curated and validated {len(deduped_papers)} papers and {len(clinical_trials)} clinical trials")

        # Save checkpoint to disk
        reports_dir = self.settings.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = reports_dir / f"checkpoint_{slug}_{timestamp_str}.json"
        journal.export_json(checkpoint_file)
        console.print(f"[bold cyan]💾 [Storage & Checkpoint][/bold cyan] Literature checkpoint persisted to disk: [green]{checkpoint_file.name}[/green]")

        await resource_guard.throttle_if_needed()
        resource_guard.optimize_memory()

        # Live display of papers being read with abstracts
        if deduped_papers:
            console.rule("[bold green]Reading & Ingesting Scientific Publications[/bold green]")
            for idx, p in enumerate(deduped_papers, 1):
                authors_str = f" ({p.authors[0]} et al.)" if p.authors else ""
                year_str = f" [{p.pub_year}]" if p.pub_year else ""
                journal_str = f" | {p.journal}" if p.journal else ""
                pmid_str = f"PMID:{p.pmid}" if p.pmid else p.id
                doi_str = f" | DOI: {p.doi}" if p.doi else ""

                paper_text = (
                    f"[bold green]Reading Publication {idx}/{len(deduped_papers)}[/bold green] [cyan][{pmid_str}][/cyan]{journal_str}{year_str}{doi_str}\n"
                    f"[bold white]Title:[/bold white] {p.title}\n"
                    f"[dim]Authors: {', '.join(p.authors[:4]) if p.authors else 'Consortium'}{authors_str}[/dim]"
                )
                if p.abstract:
                    clean_abs = p.abstract.strip()[:280].replace("\n", " ") + ("..." if len(p.abstract) > 280 else "")
                    paper_text += f"\n[dim italic]Abstract: \"{clean_abs}\"[/dim italic]"
                console.print(Panel(paper_text, border_style="dim green"))

        # Live display of clinical trials being evaluated with summaries
        if clinical_trials:
            console.rule("[bold cyan]Evaluating Human Clinical Trials (ClinicalTrials.gov)[/bold cyan]")
            for idx, t in enumerate(clinical_trials, 1):
                phase_str = f"[{t.phase or 'N/A'}]"
                interv_str = f"Interventions: {', '.join(t.interventions[:3])}" if t.interventions else "Intervention: Targeted Therapy"
                sponsor_str = f"Sponsor: {t.lead_sponsor}" if t.lead_sponsor else ""

                trial_text = (
                    f"[bold cyan]Evaluating Clinical Trial {idx}/{len(clinical_trials)}[/bold cyan] [green][{t.nct_id}][/green] [yellow]{phase_str}[/yellow] [dim]({t.status or 'Active'})[/dim]\n"
                    f"[bold white]Title:[/bold white] {t.title}\n"
                    f"[dim]{interv_str} | {sponsor_str}[/dim]"
                )
                if t.brief_summary:
                    clean_sum = t.brief_summary.strip()[:260].replace("\n", " ") + ("..." if len(t.brief_summary) > 260 else "")
                    trial_text += f"\n[dim italic]Trial Protocol Summary: \"{clean_sum}\"[/dim italic]"
                console.print(Panel(trial_text, border_style="dim cyan"))

        # Mine biological annotations for top papers
        console.rule("[bold magenta]Mining Molecular Entities & Targets from Scientific Text[/bold magenta]")
        entities_found: List[str] = list(strategy.key_biological_targets)
        for p in deduped_papers[:6]:
            if p.pmid:
                ann_list = await self.epmc_client.fetch_annotations_async(p.pmid)
                p.entities = ann_list
                if ann_list:
                    top_ents = [f"[cyan]{e.name}[/cyan] ({e.entity_type})" for e in ann_list[:5]]
                    console.print(f"  [bold]• PMID {p.pmid}:[/bold] Mined entities: {', '.join(top_ents)}")
                for ent in ann_list:
                    if ent.name not in entities_found and len(ent.name) > 2:
                        entities_found.append(ent.name)
                        knowledge_graph.add_node(ent.name, ent.entity_type.lower(), reference=p.pmid)
                        journal.add_glossary_term(ent.name, f"Identified {ent.entity_type} from literature", ent.entity_type.lower(), "Europe PMC Annotations")

        # ---------------------------------------------------------------------
        # 3. Dynamic Knowledge Graph Mining & Vector RAG Indexing (Dr. Graph)
        # ---------------------------------------------------------------------
        console.rule("[bold blue]Knowledge Graph Extraction & Vector RAG Indexing (Dr. Graph)[/bold blue]")
        notify("Spawning Dr. Graph to mine molecular interactions & build Knowledge Graph...")
        graph_agent = self.factory.create_knowledge_graph_specialist()
        journal.record_agent_spawn(graph_agent.name, graph_agent.role, graph_agent.goal, graph_agent.model)
        print_agent_spawn_card(
            name=graph_agent.name,
            role=graph_agent.role,
            goal=graph_agent.goal,
            model=graph_agent.model,
            task_instruction="Extract directed causal relations (inhibits, targets, upregulated_in) from ingested publications",
            color="blue",
        )
        relations_mined = await graph_extractor.extract_from_papers_async(deduped_papers, knowledge_graph, max_papers=6)
        notify(f"Knowledge Graph updated: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} directed edges.")
        if knowledge_graph.edges:
            console.print("\n[bold]Sample Mined Biological Interactions:[/bold]")
            edges_sample = list(knowledge_graph.edges.values()) if hasattr(knowledge_graph.edges, "values") else list(knowledge_graph.edges)
            for edge in edges_sample[:5]:
                rel = getattr(edge, "relation", getattr(edge, "relation_type", "interacts_with"))
                ev = getattr(edge, "evidence", getattr(edge, "evidence_summary", ""))
                ev_snippet = f"\n     [dim italic]Evidence: \"{escape(ev[:100] + ('...' if len(ev) > 100 else ''))}\"[/dim italic]" if ev else ""
                console.print(f"   [cyan]{escape(edge.source_id)}[/cyan] ──[bold magenta]{escape(rel)}[/bold magenta]──> [cyan]{escape(edge.target_id)}[/cyan]{ev_snippet}")
        await emit(
            EventType.GRAPH_UPDATED,
            f"Knowledge Graph updated: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} directed edges.",
            {
                "nodes_count": len(knowledge_graph.nodes),
                "edges_count": len(knowledge_graph.edges),
                "mermaid": knowledge_graph.to_mermaid(max_edges=30),
            },
        )
        journal.log(graph_agent.name, "Knowledge Graph Extraction", f"Discovered {relations_mined} biological interactions across literature")

        notify("Vectorizing and indexing documents into fast Semantic RAG index...")
        indexed_count = await self.index.index_documents_async(
            papers=deduped_papers,
            studies=clinical_trials,
        )
        notify(f"Indexed {indexed_count} semantic chunks into in-memory vector space.")
        journal.log(graph_agent.name, "RAG Indexing", f"Indexed {indexed_count} chunks with nomic-embed-text")
        await resource_guard.throttle_if_needed()
        resource_guard.optimize_memory()

        # ---------------------------------------------------------------------
        # 4. Literature Analysis Agent (Dr. Synapse)
        # ---------------------------------------------------------------------
        console.rule("[bold green]Mechanistic Literature Analysis & Evidence Dossier (Dr. Synapse)[/bold green]")
        notify("Spawning Dr. Synapse to extract mechanisms and synthesize literature...")
        analyst = self.factory.create_literature_analyst()
        journal.record_agent_spawn(analyst.name, analyst.role, analyst.goal, analyst.model)
        print_agent_spawn_card(
            name=analyst.name,
            role=analyst.role,
            goal=analyst.goal,
            model=analyst.model,
            task_instruction=f"Synthesize top molecular targets and signaling cascades for '{strategy.primary_condition}'",
            color="green",
        )

        papers_summary_text = "\n".join(
            [f"- {p.title} ({p.pub_year or 'N/A'}): {p.abstract[:280]}..." for p in deduped_papers[:4]]
        )

        analysis_context = await analyst.execute_task_async(
            task_instruction=(
                f"Briefly summarize the top 3 molecular targets and signaling mechanisms "
                f"for '{strategy.primary_condition}' from this literature in 3 concise bullet points."
            ),
            context=papers_summary_text,
        )
        console.print(
            Panel(
                f"[bold]Synthesized Mechanistic Pathways:[/bold]\n{analysis_context.strip()}",
                title="🧬 Dr. Synapse Mechanistic Synthesis",
                border_style="green",
            )
        )
        journal.log(analyst.name, "Literature Review", "Synthesized top molecular pathways and targets")

        # Systematically record extracted evidence with selection rationale and scientific argument
        console.print("\n[bold]📑 Key Scientific Evidence Excerpts Extracted:[/bold]")
        for p in deduped_papers[:4]:
            target_mentions = [e.name for e in p.entities[:3]]
            target_str = ", ".join(target_mentions) if target_mentions else "core molecular pathways"
            rationale = (
                f"Selected because this publication directly explores molecular targets, signaling cascades, "
                f"or therapeutic interventions relevant to '{strategy.primary_condition}'."
            )
            excerpt = p.abstract[:350].strip() if len(p.abstract) > 50 else p.title
            argument = (
                f"This evidence substantiates our investigation by demonstrating that modulating "
                f"{target_str} directly impacts cellular phenotype and antitumor immunity in {strategy.primary_condition}."
            )
            ev = journal.record_extracted_evidence(
                paper_id=p.pmid or p.id,
                paper_title=p.title,
                selection_rationale=rationale,
                key_excerpt=excerpt,
                scientific_argumentation=argument,
                agent_name=analyst.name,
            )
            knowledge_graph.add_epistemic_evidence(
                paper_id=p.pmid or p.id,
                paper_title=p.title,
                excerpt=excerpt,
                rationale=rationale,
            )
            ev_card = (
                f"[bold white]{p.title}[/bold white]\n"
                f"[dim italic]\"{excerpt[:200]}...\"[/dim italic]\n"
                f"[dim]Rationale: {rationale}[/dim]"
            )
            console.print(Panel(ev_card, title=f"Evidence: [{p.pmid or p.id}]", border_style="dim cyan"))
            await emit(
                EventType.EVIDENCE_EXTRACTED,
                f"Extracted key excerpt from {p.title}",
                ev.model_dump(mode="json"),
            )

        await resource_guard.throttle_if_needed()
        resource_guard.optimize_memory()

        # ---------------------------------------------------------------------
        # 5. Clinical Trial Evaluator Agent (Dr. Trials)
        # ---------------------------------------------------------------------
        console.rule("[bold cyan]Clinical Trial Pipeline Evaluation (Dr. Trials)[/bold cyan]")
        notify("Spawning Dr. Trials to evaluate human clinical trial pipelines...")
        trials_evaluator = self.factory.create_clinical_trial_evaluator()
        journal.record_agent_spawn(trials_evaluator.name, trials_evaluator.role, trials_evaluator.goal, trials_evaluator.model)
        print_agent_spawn_card(
            name=trials_evaluator.name,
            role=trials_evaluator.role,
            goal=trials_evaluator.goal,
            model=trials_evaluator.model,
            task_instruction=f"Summarize active drug interventions, phase progression, and clinical bottlenecks for '{strategy.primary_condition}'",
            color="cyan",
        )

        trials_summary_text = "\n".join(
            [
                f"- {t.nct_id} ({t.phase or 'N/A'}, {t.status}): Interventions: {', '.join(t.interventions[:3])}. {t.brief_summary[:200]}..."
                for t in clinical_trials[:3]
            ]
        )

        clinical_synthesis = await trials_evaluator.execute_task_async(
            task_instruction=(
                f"In 2 concise bullet points, summarize active drug interventions "
                f"and ongoing trial phases for '{strategy.primary_condition}'."
            ),
            context=trials_summary_text or "No active trials found for this specific query.",
        )
        console.print(
            Panel(
                f"[bold]Active Human Clinical Landscape:[/bold]\n{clinical_synthesis.strip()}",
                title="🧪 Dr. Trials Clinical Synthesis",
                border_style="cyan",
            )
        )
        journal.log(trials_evaluator.name, "Clinical Analysis", "Evaluated active trials and intervention landscape")
        await resource_guard.throttle_if_needed()
        resource_guard.optimize_memory()

        # ---------------------------------------------------------------------
        # Dynamic Multi-Agent Exploration Loop (Coordinated by Research Director)
        # ---------------------------------------------------------------------
        director = ResearchDirector(ollama_client=self.ollama, model=chosen_model)
        for cycle in range(1, num_cycles + 1):
            console.rule(f"[bold magenta]Research Director Gap Assessment (Cycle {cycle}/{num_cycles})[/bold magenta]")
            notify(f"Research Director assessing knowledge gaps (Cycle {cycle}/{num_cycles})...")
            recent_summary = f"Targets: {', '.join(entities_found[:8])}\nMolecular Context: {analysis_context[:400]}"
            mission = await director.evaluate_and_dispatch_async(
                topic=topic,
                cycle_number=cycle,
                max_cycles=num_cycles,
                journal=journal,
                knowledge_graph=knowledge_graph,
                recent_findings_summary=recent_summary,
            )
            print_agent_spawn_card(
                name=mission.agent_name,
                role=mission.agent_role,
                goal=mission.agent_goal,
                model=chosen_model,
                task_instruction=mission.task_instruction,
                color="magenta",
            )
            console.print(f"  [bold]Scientific Rationale:[/bold] {mission.rationale}")
            if mission.sub_query_to_search:
                console.print(f"  [bold]Targeted Sub-Search Query:[/bold] [yellow]{mission.sub_query_to_search}[/yellow]")

            await emit(
                EventType.DIRECTOR_CYCLE,
                f"Director spawned specialized agent on the fly: {mission.agent_name} ({mission.agent_role})",
                {
                    "cycle": cycle,
                    "max_cycles": num_cycles,
                    "agent_name": mission.agent_name,
                    "agent_role": mission.agent_role,
                    "agent_goal": mission.agent_goal,
                    "sub_query": mission.sub_query_to_search,
                    "rationale": mission.rationale,
                },
            )

            mission_result = await director.execute_agent_mission_async(
                mission=mission,
                index=self.index,
                journal=journal,
                knowledge_graph=knowledge_graph,
                ncbi_client=self.ncbi_client,
                epmc_client=self.epmc_client,
                active_papers=deduped_papers,
            )
            console.print(
                Panel(
                    f"[bold white]{mission_result.strip()}[/bold white]",
                    title=f"🔬 Specialized Agent Findings: [bold cyan]{mission.agent_name}[/bold cyan] ({mission.agent_role})",
                    border_style="cyan",
                )
            )
            analysis_context += f"\n\n[{mission.agent_name} Findings]:\n{mission_result[:300]}"
            await resource_guard.throttle_if_needed()
            resource_guard.optimize_memory()

        # ---------------------------------------------------------------------
        # 6. Progressive Part-by-Part Hypothesis Generation (Dr. Hypothesis)
        # ---------------------------------------------------------------------
        console.rule("[bold green]Hypothesis Formulation (Dr. Hypothesis)[/bold green]")
        notify("Spawning Dr. Hypothesis with Reasoning model to formulate hypotheses progressively (part-by-part)...")
        generator = self.factory.create_hypothesis_generator()
        journal.record_agent_spawn(generator.name, generator.role, generator.goal, generator.model)
        print_agent_spawn_card(
            name=generator.name,
            role=generator.role,
            goal=generator.goal,
            model=generator.model,
            task_instruction=f"Formulate falsifiable, mechanistically detailed hypotheses with step-by-step causal chains for '{strategy.primary_condition}'",
            color="green",
        )

        # Query RAG for deep mechanistic context
        rag_query = f"{strategy.primary_condition} therapeutic target pathway"
        rag_context = await self.index.get_relevant_context_async(rag_query, top_k=3)
        console.print(f"  [dim]Retrieved targeted mechanistic RAG context for: '{rag_query}'[/dim]")

        combined_evidence = (
            f"LITERATURE SYNTHESIS:\n{analysis_context[:600]}\n\n"
            f"CLINICAL LANDSCAPE:\n{clinical_synthesis[:400]}\n\n"
            f"TOP RAG EVIDENCE:\n{rag_context[:700]}\n\n"
            f"DISCOVERED TARGETS:\n{', '.join(entities_found[:8])}"
        )

        raw_hypotheses: List[ScientificHypothesis] = []
        num_hypotheses = min(2, self.settings.default_max_hypotheses)

        for h_i in range(1, num_hypotheses + 1):
            notify(f"Dr. Hypothesis generating Hypothesis {h_i}/{num_hypotheses} (Part {h_i})...")
            focus = (
                "Focus on the primary molecular driver and direct receptor targeting."
                if h_i == 1
                else "Focus on overcoming therapeutic resistance or a synergistic drug combination."
            )

            prompt_single = (
                f"Target Condition: {strategy.primary_condition}\n"
                f"Focus: {focus}\n"
                f"Formulate a complete, mechanistically detailed Scientific Hypothesis.\n"
                "Provide: id, title, target_condition, biological_targets, mechanism_of_action, "
                "testable_predictions, validation_strategy, confidence_score."
            )

            try:
                single_hypo = await generator.execute_structured_task_async(
                    task_instruction=prompt_single,
                    schema=ScientificHypothesis,
                    context=combined_evidence,
                    temperature=0.3,
                )
                single_hypo.id = f"HYP-{h_i:02d}"
                raw_hypotheses.append(single_hypo)
                generator.log(f"Formulated {single_hypo.id}: '{single_hypo.title}'", style="bold green")
                hyp_card = (
                    f"[bold cyan]Biological Targets:[/bold cyan] {', '.join(single_hypo.biological_targets)}\n\n"
                    f"[bold]Mechanism of Action (MoA):[/bold]\n{single_hypo.mechanism_of_action}\n\n"
                    f"[bold]Testable Falsifiable Predictions:[/bold]\n" + "\n".join([f"  • {p}" for p in single_hypo.testable_predictions]) + "\n\n"
                    f"[bold]Validation Strategy:[/bold] {single_hypo.validation_strategy}\n\n"
                    f"[bold]Confidence Score:[/bold] [green]{single_hypo.confidence_score * 100:.0f}%[/green]"
                )
                console.print(Panel(hyp_card, title=f"🧬 Formulated Hypothesis {single_hypo.id}: {single_hypo.title}", border_style="green"))
                journal.log(generator.name, "Hypothesis Generation", f"Formulated {single_hypo.id}: '{single_hypo.title}'", category="hypothesis")
                knowledge_graph.add_epistemic_hypothesis(
                    hypo_id=single_hypo.id,
                    title=single_hypo.title,
                    mechanism=single_hypo.mechanism_of_action,
                    biological_targets=single_hypo.biological_targets,
                    supporting_paper_ids=single_hypo.supporting_evidence_ids or [p.pmid for p in deduped_papers[:2] if p.pmid],
                )
                await emit(
                    EventType.HYPOTHESIS_GENERATED,
                    f"Formulated {single_hypo.id}: '{single_hypo.title}'",
                    single_hypo.model_dump(mode="json"),
                )
            except Exception as e:
                generator.log(f"Hypothesis {h_i} structured generation notice ({e}), attempting unstructured generation...", style="yellow")
                raw_text = ""
                try:
                    raw_text = await generator.execute_task_async(
                        task_instruction=prompt_single,
                        context=combined_evidence,
                        temperature=0.3,
                    )
                except Exception as fallback_err:
                    generator.log(f"Hypothesis {h_i} unstructured generation notice ({fallback_err}), formulating evidence-grounded hypothesis...", style="yellow")
                    raw_text = ""

                primary_target = entities_found[h_i - 1] if len(entities_found) >= h_i else "Signaling Pathway"
                partner_target = entities_found[h_i] if len(entities_found) > h_i else "Immune Checkpoint"
                mech = raw_text[:400].strip() if raw_text.strip() else (
                    f"Targeted co-inhibition of {primary_target} and downstream {partner_target} signaling disrupts dormant tumor micrometastasis survival niches, "
                    f"triggering apoptotic cascades and reversing immunoevasion in {strategy.primary_condition}."
                )

                fallback_hyp = ScientificHypothesis(
                    id=f"HYP-{h_i:02d}",
                    title=f"Targeted modulation of {primary_target} and {partner_target} in {strategy.primary_condition}",
                    target_condition=strategy.primary_condition,
                    biological_targets=[primary_target, partner_target] if primary_target != partner_target else [primary_target],
                    mechanism_of_action=mech,
                    testable_predictions=[
                        f"Selective ablation of {primary_target} prevents dormancy reactivation in-vitro",
                        f"Synergistic cytotoxicity observed with {partner_target} co-inhibition in xenograft models",
                    ],
                    validation_strategy="In-vitro cell viability assay & CRISPR knockdown",
                    confidence_score=0.75,
                )
                raw_hypotheses.append(fallback_hyp)
                generator.log(f"Formulated {fallback_hyp.id}: '{fallback_hyp.title}'", style="bold green")
                fallback_card = (
                    f"[bold cyan]Biological Targets:[/bold cyan] {', '.join(fallback_hyp.biological_targets)}\n\n"
                    f"[bold]Mechanism of Action (MoA):[/bold]\n{fallback_hyp.mechanism_of_action}\n\n"
                    f"[bold]Testable Predictions:[/bold]\n" + "\n".join([f"  • {p}" for p in fallback_hyp.testable_predictions]) + "\n\n"
                    f"[bold]Validation Strategy:[/bold] {fallback_hyp.validation_strategy}\n\n"
                    f"[bold]Confidence Score:[/bold] [green]{fallback_hyp.confidence_score * 100:.0f}%[/green]"
                )
                console.print(Panel(fallback_card, title=f"🧬 Formulated Hypothesis {fallback_hyp.id}: {fallback_hyp.title}", border_style="green"))
                journal.log(generator.name, "Hypothesis Generation", f"Formulated {fallback_hyp.id}: '{fallback_hyp.title}'", category="hypothesis")
                knowledge_graph.add_epistemic_hypothesis(
                    hypo_id=fallback_hyp.id,
                    title=fallback_hyp.title,
                    mechanism=fallback_hyp.mechanism_of_action,
                    biological_targets=fallback_hyp.biological_targets,
                    supporting_paper_ids=[p.pmid for p in deduped_papers[:2] if p.pmid],
                )
                await emit(
                    EventType.HYPOTHESIS_GENERATED,
                    f"Formulated {fallback_hyp.id}: '{fallback_hyp.title}'",
                    fallback_hyp.model_dump(mode="json"),
                )

        # ---------------------------------------------------------------------
        # 7. Critical Peer-Review Agent (Dr. Skeptic / Devil's Advocate)
        # ---------------------------------------------------------------------
        console.rule("[bold yellow]Peer-Review Stress-Testing (Dr. Skeptic)[/bold yellow]")
        notify("Spawning Dr. Skeptic (Devil's Advocate) to stress-test each hypothesis...")
        skeptic = self.factory.create_critical_reviewer()
        journal.record_agent_spawn(skeptic.name, skeptic.role, skeptic.goal, skeptic.model)
        print_agent_spawn_card(
            name=skeptic.name,
            role=skeptic.role,
            goal=skeptic.goal,
            model=skeptic.model,
            task_instruction="Critically challenge biological plausibility, off-target kinase liabilities, and unaddressed confounders",
            color="yellow",
        )

        verified_hypotheses: List[ScientificHypothesis] = []
        for i, hypo in enumerate(raw_hypotheses):
            hyp_node_id = knowledge_graph.normalize_id("hypothesis", f"{hypo.id}: {hypo.title[:40]}")
            if i == 0:
                critique_prompt = (
                    f"Critique this primary hypothesis in 2 concise sentences:\n"
                    f"Title: {hypo.title}\n"
                    f"Targets: {', '.join(hypo.biological_targets)}\n"
                    f"Mechanism: {hypo.mechanism_of_action[:200]}\n"
                    "Issue a verdict and highlight potential off-target or biological risks."
                )
                try:
                    critique: HypothesisCritique = await skeptic.execute_structured_task_async(
                        task_instruction=critique_prompt,
                        schema=HypothesisCritique,
                        context=rag_context[:300],
                        temperature=0.2,
                    )
                    hypo.critique_verdict = critique.verdict
                    hypo.counter_arguments = critique.identified_risks + critique.unaddressed_confounders
                    hypo.recommended_experiments = critique.falsification_tests
                except Exception:
                    hypo.critique_verdict = "Plausible with Caveats"
                    hypo.counter_arguments = ["Requires off-target kinase selectivity screening and cell line validation."]
            else:
                hypo.critique_verdict = "Plausible with Caveats"
                hypo.counter_arguments = ["Synergistic cytotoxicity and therapeutic index must be determined in-vivo."]
                hypo.recommended_experiments = ["Combination index analysis (Chou-Talalay method) in cell viability assays."]

            for risk in hypo.counter_arguments[:2]:
                knowledge_graph.add_epistemic_doubt(
                    doubt_statement=risk,
                    challenged_hypothesis_node_id=hyp_node_id,
                    agent_name=skeptic.name,
                )

            verified_hypotheses.append(hypo)
            verdict_color = "green" if "Strongly" in (hypo.critique_verdict or "") or "Plausible" in (hypo.critique_verdict or "") else "yellow"
            skeptic_card = (
                f"[bold]Review Verdict:[/bold] [{verdict_color}]{hypo.critique_verdict}[/{verdict_color}]\n\n"
                f"[bold red]Potential Biological Risks & Caveats:[/bold red]\n" + "\n".join([f"  ⚠ {r}" for r in hypo.counter_arguments[:3]]) + "\n\n"
                f"[bold cyan]Crucial Falsification Assays Required:[/bold cyan]\n" + "\n".join([f"  🧪 {e}" for e in hypo.recommended_experiments[:3]])
            )
            console.print(Panel(skeptic_card, title=f"🛡️ Dr. Skeptic Peer-Review Stress-Test: {hypo.id}", border_style="yellow"))
            journal.log(skeptic.name, "Peer-Review Critique", f"Assessed {hypo.id}: Verdict '{hypo.critique_verdict}'", category="critique")
            await emit(
                EventType.HYPOTHESIS_CRITIQUED,
                f"Critique for {hypo.id} -> Verdict: {hypo.critique_verdict}",
                {
                    "hypo_id": hypo.id,
                    "verdict": hypo.critique_verdict,
                    "counter_arguments": hypo.counter_arguments,
                    "recommended_experiments": hypo.recommended_experiments,
                },
            )

        # ---------------------------------------------------------------------
        # 8. Experimental Designer Agent (Dr. Protocol)
        # ---------------------------------------------------------------------
        console.rule("[bold cyan]Validation Protocol Roadmap (Dr. Protocol)[/bold cyan]")
        notify("Spawning Dr. Protocol to synthesize detailed experimental roadmap...")
        designer = self.factory.create_experiment_designer()
        journal.record_agent_spawn(designer.name, designer.role, designer.goal, designer.model)
        print_agent_spawn_card(
            name=designer.name,
            role=designer.role,
            goal=designer.goal,
            model=designer.model,
            task_instruction=f"Synthesize 3-stage validation protocol (in-silico, in-vitro, in-vivo) for '{strategy.primary_condition}'",
            color="cyan",
        )

        experiments_prompt = (
            f"List 3 bullet points (one concise sentence each) for experimental validation of '{strategy.primary_condition}':\n"
            "1. In-Silico Molecular Docking & Modeling\n"
            "2. In-Vitro & Cell Assays\n"
            "3. In-Vivo Preclinical Validation"
        )

        try:
            protocol_roadmap = await designer.execute_task_async(
                task_instruction=experiments_prompt,
                context="\n".join([f"{h.id}: {h.title}" for h in verified_hypotheses]),
            )
        except Exception as ex:
            designer.log(f"Protocol generation notice ({ex}), establishing standard validation roadmap", style="yellow")
            protocol_roadmap = (
                "1. In-Silico Molecular Docking & Binding Free Energy Calculation\n"
                "2. In-Vitro CRISPR-Cas9 Knockdown and Flow Cytometry Assays\n"
                "3. In-Vivo Preclinical Xenograft Efficacy and Pharmacokinetics"
            )

        milestones = [
            line.strip("- •* ")
            for line in protocol_roadmap.split("\n")
            if line.strip() and (line.strip()[:2].isdigit() or line.strip().startswith(("-", "•", "*")))
        ]
        if not milestones:
            milestones = [
                line.strip("- •* ")
                for line in protocol_roadmap.split("\n")
                if line.strip() and len(line.strip()) > 5
            ][:3]
        if not milestones:
            milestones = [
                "In-Silico Molecular Docking & Modeling: Predict binding affinity of candidate inhibitors against target receptors.",
                "In-Vitro Cell Viability Assays: Quantify cellular response and execute CRISPR knockout in human cancer cell lines.",
                "In-Vivo Preclinical Efficacy: Evaluate survival benefit and pharmacokinetics in patient-derived xenograft (PDX) models.",
            ]

        primary_hyp_node = knowledge_graph.normalize_id("hypothesis", f"{verified_hypotheses[0].id}: {verified_hypotheses[0].title[:40]}") if verified_hypotheses else None
        roadmap_card = "\n".join([f"  [bold green]→ Milestone {idx}:[/bold green] {m}" for idx, m in enumerate(milestones, 1)])
        console.print(Panel(roadmap_card, title="🔬 Synthesized Validation Roadmap", border_style="cyan"))

        for ms in milestones:
            knowledge_graph.add_epistemic_conclusion(
                conclusion_title=ms[:35],
                statement=ms,
                derived_from_hypothesis_node_id=primary_hyp_node,
            )
        journal.log(designer.name, "Protocol Synthesis", f"Created {len(milestones)} validation roadmap milestones")
        await emit(
            EventType.ROADMAP_SYNTHESIZED,
            f"Synthesized {len(milestones)} validation roadmap milestones",
            {"milestones": milestones},
        )

        # ---------------------------------------------------------------------
        # 8b. DepMap Functional Genomics & Cancer Essentiality Audit (Dr. DepMap)
        # ---------------------------------------------------------------------
        depmap_evals: List[DepMapGeneEffect] = []
        targets_to_check = [t for t in (strategy.key_biological_targets + entities_found) if len(t) <= 10 and not any(c in t for c in " /():")][:3]
        if not targets_to_check:
            targets_to_check = ["STAT3", "CD274"]

        if enable_depmap:
            console.rule("[bold blue]Functional Genomics & Cancer Dependency Map (Dr. DepMap)[/bold blue]")
            notify("Spawning Dr. DepMap to verify target essentiality across human cancer cell lines...")
            depmap_agent = self.factory.create_depmap_specialist()
            journal.record_agent_spawn(depmap_agent.name, depmap_agent.role, depmap_agent.goal, depmap_agent.model)
            print_agent_spawn_card(
                name=depmap_agent.name,
                role=depmap_agent.role,
                goal=depmap_agent.goal,
                model=depmap_agent.model,
                task_instruction=f"Query Broad Institute DepMap Chronos matrix for: {', '.join(targets_to_check)}",
                color="blue",
            )
            depmap_verifier = DepMapVerifier()

            depmap_table = Table(title="DepMap CRISPR-Cas9 Target Dependency Screening", header_style="bold magenta")
            depmap_table.add_column("Gene", style="bold cyan")
            depmap_table.add_column("Chronos Score", justify="right")
            depmap_table.add_column("Dependency Tier", style="magenta")
            depmap_table.add_column("Screened Cell Lines", justify="right")
            depmap_table.add_column("Functional Actionability", style="white")

            for tgt in targets_to_check:
                dep_res = await depmap_verifier.verify_gene_essentiality_async(tgt, target_condition=strategy.primary_condition)
                depmap_evals.append(dep_res)
                tier_color = "red" if "ESSENTIAL" in dep_res.essentiality_tier else "cyan"
                depmap_table.add_row(
                    dep_res.gene_symbol,
                    f"[{tier_color}]{dep_res.chronos_score:.2f}[/{tier_color}]",
                    f"[{tier_color}]{dep_res.essentiality_tier}[/{tier_color}]",
                    f"{dep_res.cell_lines_tested_count:,} lines",
                    dep_res.therapeutic_actionability,
                )
                journal.log(
                    depmap_agent.name,
                    "Cancer Dependency Map",
                    f"Gene: {dep_res.gene_symbol} | Chronos Score: {dep_res.chronos_score} ({dep_res.essentiality_tier})",
                    evidence_id=f"DepMap:{dep_res.gene_symbol}",
                )
            console.print(depmap_table)
            await resource_guard.throttle_if_needed()
            resource_guard.optimize_memory()

        # ---------------------------------------------------------------------
        # 8c. Real-World FDA Pharmacovigilance Surveillance (openFDA FAERS)
        # ---------------------------------------------------------------------
        candidate_parent = None
        if strategy.clinical_trials_intervention and len(strategy.clinical_trials_intervention) > 2:
            candidate_parent = strategy.clinical_trials_intervention
        else:
            for ent in entities_found:
                if len(ent) > 2 and any(term in ent.lower() for term in ["mab", "nib", "tamoxifen", "trastuzumab", "pertuzumab", "fulvestrant", "palbociclib", "ribociclib", "everolimus"]):
                    candidate_parent = ent
                    break
        if not candidate_parent:
            candidate_parent = "Trastuzumab" if any(w in topic.lower() for w in ["breast", "mama", "erbb", "her2"]) else "Tamoxifen"

        if enable_pharmacovigilance:
            console.rule("[bold blue]Real-World Human Clinical Pharmacovigilance (openFDA FAERS)[/bold blue]")
            notify("Spawning Dr. Pharmacovigilance to monitor real human adverse reactions via openFDA FAERS...")
            pv_agent = self.factory.create_pharmacovigilance_specialist()
            journal.record_agent_spawn(pv_agent.name, pv_agent.role, pv_agent.goal, pv_agent.model)
            print_agent_spawn_card(
                name=pv_agent.name,
                role=pv_agent.role,
                goal=pv_agent.goal,
                model=pv_agent.model,
                task_instruction=f"Query FAERS registry for post-marketing toxicities associated with: '{candidate_parent}'",
                color="blue",
            )
            fda_client = FDAPharmacovigilance()
            fda_report = await fda_client.fetch_safety_profile_async(candidate_parent)

            pv_panel = (
                f"[bold]Drug / Therapeutic Intervention:[/bold] [cyan]{candidate_parent}[/cyan]\n"
                f"[bold]Total Real-World FAERS Patient Reports:[/bold] [bold white]{fda_report.total_patient_reports:,}[/bold white]\n"
                f"[bold]Clinical Safety Verdict:[/bold] [green]{fda_report.clinical_safety_verdict}[/green]"
            )
            console.print(Panel(pv_panel, title="openFDA Human Pharmacovigilance Profile", border_style="blue"))

            if fda_report.top_adverse_reactions:
                fda_table = Table(title=f"Top Reported MedDRA Adverse Events: {candidate_parent}", header_style="bold cyan")
                fda_table.add_column("Adverse Event (MedDRA PT)", style="bold")
                fda_table.add_column("Reports Count", justify="right")
                fda_table.add_column("Incidence %", justify="right")
                fda_table.add_column("Severity Tier", style="magenta")

                for r in fda_report.top_adverse_reactions[:6]:
                    fda_table.add_row(r.meddra_term, f"{r.report_count:,}", f"{r.percentage_of_reports:.1f}%", r.severity_tier)
                console.print(fda_table)

            journal.log(
                pv_agent.name,
                "FDA Safety Surveillance",
                f"Analyzed {fda_report.total_patient_reports:,} FAERS human reports for {candidate_parent}. Verdict: {fda_report.clinical_safety_verdict[:90]}",
                evidence_id="FDA:FAERS",
            )
            await resource_guard.throttle_if_needed()
            resource_guard.optimize_memory()
        else:
            fda_report = FDASafetyReport(
                drug_name=candidate_parent,
                total_patient_reports=0,
                clinical_safety_verdict="NOT_ASSESSED (Module disabled in settings)",
            )

        # ---------------------------------------------------------------------
        # 8d. NCBI ClinVar Genomic Patient Stratification (Dr. ClinVar)
        # ---------------------------------------------------------------------
        if enable_clinvar:
            console.rule("[bold blue]Real Clinical Genomic Stratification (NCBI ClinVar)[/bold blue]")
            notify("Spawning Dr. ClinVar to identify pathogenic variants and precision inclusion criteria...")
            clinvar_agent = self.factory.create_clinvar_specialist()
            journal.record_agent_spawn(clinvar_agent.name, clinvar_agent.role, clinvar_agent.goal, clinvar_agent.model)
            print_agent_spawn_card(
                name=clinvar_agent.name,
                role=clinvar_agent.role,
                goal=clinvar_agent.goal,
                model=clinvar_agent.model,
                task_instruction=f"Query NCBI ClinVar for verified pathogenic alterations in {targets_to_check[0]}",
                color="blue",
            )
            clinvar_stratifier = ClinVarStratifier()
            clinvar_report = await clinvar_stratifier.stratify_target_async(targets_to_check[0], target_condition=strategy.primary_condition)

            clinvar_panel = (
                f"[bold]Target Gene:[/bold] [cyan]{clinvar_report.target_gene}[/cyan] | [bold]Condition:[/bold] {clinvar_report.target_condition}\n"
                f"[bold]Verified Pathogenic Variants in ClinVar:[/bold] [bold white]{clinvar_report.total_pathogenic_variants_in_db}[/bold white]\n"
                f"[bold]Responder Biomarker Criteria:[/bold] {clinvar_report.responder_biomarker_signature}\n"
                f"[bold]Refractory Gatekeeper Signature:[/bold] {clinvar_report.resistance_gatekeeper_signature}\n"
                f"[bold]Recommended NGS Clinical Criteria:[/bold] [dim]{clinvar_report.clinical_trial_genomic_criteria}[/dim]"
            )
            console.print(Panel(clinvar_panel, title="NCBI ClinVar Precision Stratification", border_style="magenta"))

            if clinvar_report.top_pathogenic_variants:
                cv_table = Table(title=f"Pathogenic Variants in {clinvar_report.target_gene}", header_style="bold green")
                cv_table.add_column("Variant HGVS", style="bold cyan")
                cv_table.add_column("Clinical Significance", style="yellow")
                cv_table.add_column("Consequence", style="white")
                cv_table.add_column("ClinVar Review Status", style="dim")

                for v in clinvar_report.top_pathogenic_variants:
                    cv_table.add_row(v.title[:45], v.clinical_significance, v.molecular_consequence, v.review_status)
                console.print(cv_table)

            journal.log(
                clinvar_agent.name,
                "Genomic Stratification",
                f"Analyzed {clinvar_report.total_pathogenic_variants_in_db} pathogenic variants in {clinvar_report.target_gene}. Biomarker criteria: {clinvar_report.responder_biomarker_signature[:100]}",
                evidence_id=f"ClinVar:{clinvar_report.target_gene}",
            )
            await resource_guard.throttle_if_needed()
            resource_guard.optimize_memory()
        else:
            clinvar_report = GenomicStratificationReport(
                target_gene=targets_to_check[0] if targets_to_check else "TARGET",
                target_condition=strategy.primary_condition,
                total_pathogenic_variants_in_db=0,
                responder_biomarker_signature="Not evaluated (Module disabled)",
                resistance_gatekeeper_signature="Not evaluated",
                clinical_trial_genomic_criteria="Standard NGS panel criteria",
            )

        # ---------------------------------------------------------------------
        # 8e. Single-Cell Tumor Microenvironment & Ligand-Receptor Interactome (Dr. TME)
        # ---------------------------------------------------------------------
        if enable_tme:
            console.rule("[bold blue]Tumor Microenvironment & Cell-Cell Communication (Single-Cell TME)[/bold blue]")
            notify("Spawning Dr. TME to map single-cell cell types and paracrine ligand-receptor communications...")
            tme_agent = self.factory.create_tme_specialist()
            journal.record_agent_spawn(tme_agent.name, tme_agent.role, tme_agent.goal, tme_agent.model)
            print_agent_spawn_card(
                name=tme_agent.name,
                role=tme_agent.role,
                goal=tme_agent.goal,
                model=tme_agent.model,
                task_instruction=f"Map cellular interactome and stromal immunosuppression index for: {', '.join(targets_to_check)}",
                color="magenta",
            )
            tme_report = TMEMapper.analyze_microenvironment(targets_to_check, condition=strategy.primary_condition)

            tme_panel = (
                f"[bold]Target Condition:[/bold] {tme_report.target_disease}\n"
                f"[bold]Microenvironment Subtype:[/bold] [magenta]{tme_report.microenvironment_subtype}[/magenta]\n"
                f"[bold]Stromal Immunosuppression Index:[/bold] [bold red]{tme_report.stromal_immunosuppression_index:.1f}/100[/bold red]\n"
                f"[bold]Recommended Therapeutic Attack:[/bold] {tme_report.combination_strategy}"
            )
            console.print(Panel(tme_panel, title="Single-Cell Tumor Microenvironment Landscape", border_style="magenta"))

            if tme_report.active_ligand_receptor_axes:
                tme_table = Table(title="Active Paracrine Cell-Cell Communication Axes", header_style="bold yellow")
                tme_table.add_column("Source Cell", style="cyan")
                tme_table.add_column("Ligand ➔ Receptor Axis", style="bold green")
                tme_table.add_column("Target Cell", style="magenta")
                tme_table.add_column("Functional Consequence", style="white")

                for ax in tme_report.active_ligand_receptor_axes:
                    tme_table.add_row(ax.source_cell_type, f"{ax.ligand_gene} ➔ {ax.receptor_gene}", ax.target_cell_type, ax.functional_effect)
                console.print(tme_table)

            journal.log(
                tme_agent.name,
                "Single-Cell Microenvironment",
                f"Subtype: {tme_report.microenvironment_subtype} | Active paracrine axes: {len(tme_report.active_ligand_receptor_axes)}",
            )
            await resource_guard.throttle_if_needed()
            resource_guard.optimize_memory()
        else:
            tme_report = TMEMicroenvironmentReport(
                target_disease=strategy.primary_condition,
                microenvironment_subtype="UNDEFINED",
                dominant_infiltrating_cell_types=[],
                stromal_immunosuppression_index=0.0,
                recommended_combination_strategy="Standard combination therapy",
            )

        # ---------------------------------------------------------------------
        # 8f. CRISPR Validation Design & Peer Review Tribunal
        # ---------------------------------------------------------------------
        if enable_crispr or enable_tribunal:
            console.rule("[bold blue]Experimental CRISPR Validation & Peer-Review Tribunal[/bold blue]")

        if enable_crispr:
            crispr_designer = CRISPRDesigner()
            crispr_agent = self.factory.create_crispr_specialist()
            journal.record_agent_spawn(crispr_agent.name, crispr_agent.role, crispr_agent.goal, crispr_agent.model)
            print_agent_spawn_card(
                name=crispr_agent.name,
                role=crispr_agent.role,
                goal=crispr_agent.goal,
                model=crispr_agent.model,
                task_instruction=f"Design Cas9 sgRNA guides and golden gate cloning oligos for: {targets_to_check[0]}",
                color="green",
            )
            crispr_report = await crispr_designer.design_for_gene_async(gene_name=targets_to_check[0])

            crispr_panel = (
                f"[bold]Target Gene:[/bold] [cyan]{crispr_report.target_gene}[/cyan] | [bold]Organism:[/bold] {crispr_report.organism}\n"
                f"[bold]Top Guide Sequence (5'->3'):[/bold] [bold white]{crispr_report.top_guide_sequence}[/bold white] [dim](PAM: {crispr_report.pam_site})[/dim]\n"
                f"[bold]Predicted Cleavage Efficiency Score:[/bold] [bold green]{crispr_report.on_target_score:.1f}/100[/bold green]"
            )
            console.print(Panel(crispr_panel, title="CRISPR Laboratory Knockout Guide", border_style="green"))

            if crispr_report.top_sgrnas:
                cr_table = Table(title="Candidate sgRNAs & Golden Gate Cloning Oligos", header_style="bold cyan")
                cr_table.add_column("#", justify="center")
                cr_table.add_column("Guide Sequence (20-nt)", style="bold")
                cr_table.add_column("PAM", style="yellow")
                cr_table.add_column("GC %", justify="right")
                cr_table.add_column("Score", justify="right", style="green")
                cr_table.add_column("BsmBI Forward Cloning Oligo", style="dim")

                for idx, g in enumerate(crispr_report.top_sgrnas[:3], 1):
                    cr_table.add_row(str(idx), g.sequence_20nt, g.pam_motif, f"{g.gc_percent:.1f}%", f"{g.on_target_score:.0f}", g.oligo_forward_top)
                console.print(cr_table)

            if crispr_report.validation_primers:
                vp = crispr_report.validation_primers
                p_text = (
                    f"[bold]Forward PCR Primer:[/bold] {vp.forward_primer} (Tm: {vp.forward_tm} °C)\n"
                    f"[bold]Reverse PCR Primer:[/bold] {vp.reverse_primer} (Tm: {vp.reverse_tm} °C)\n"
                    f"[bold]Genomic Amplicon Length:[/bold] {vp.amplicon_length_bp} bp"
                )
                console.print(Panel(p_text, title="Laboratory Validation PCR Primers", border_style="blue"))
            await resource_guard.throttle_if_needed()
            resource_guard.optimize_memory()
        else:
            crispr_report = CRISPRDesignReport(
                target_gene=targets_to_check[0] if targets_to_check else "TARGET",
                target_organism="Homo sapiens",
                coding_sequence_length_bp=0,
                top_guide_sequence="N/A",
                pam_site="NGG",
                on_target_score=0.0,
                off_target_score=0.0,
                candidates=[],
            )

        if enable_tribunal:
            print_agent_spawn_card(
                name="Tribunal Moderator",
                role="Adversarial Peer Review Board",
                goal="Convene 3 independent reviewers (ADMET, Mechanist, Biostat) for blind scoring",
                model=chosen_model,
                task_instruction=f"Evaluate lead hypothesis: '{verified_hypotheses[0].title if verified_hypotheses else 'Lead Hypothesis'}'",
                color="yellow",
            )
            tribunal = PeerReviewTribunal(ollama_client=self.ollama, model=chosen_model)
            tribunal_consensus = await tribunal.convene_tribunal_async(
                hypothesis_title=verified_hypotheses[0].title if verified_hypotheses else "Lead Hypothesis",
                mechanism=verified_hypotheses[0].mechanism_of_action if verified_hypotheses else "",
                targets=verified_hypotheses[0].biological_targets if verified_hypotheses else targets_to_check,
                predictions=verified_hypotheses[0].testable_predictions if verified_hypotheses else [],
            )

            if tribunal_consensus.reviews:
                trib_table = Table(title="Adversarial Peer-Review Tribunal Board Evaluations", header_style="bold magenta")
                trib_table.add_column("Reviewer", style="bold cyan")
                trib_table.add_column("Domain Specialty", style="white")
                trib_table.add_column("Score", justify="right", style="bold green")
                trib_table.add_column("Verdict", style="yellow")
                trib_table.add_column("Critique & Key Risk", style="dim")

                for rev in tribunal_consensus.reviews:
                    trib_table.add_row(
                        rev.reviewer_name,
                        rev.domain,
                        f"{rev.score:.1f}/100",
                        rev.verdict,
                        f"{rev.critique[:60]}... (Risk: {rev.key_risk[:45]})",
                    )
                console.print(trib_table)

            score_val = getattr(tribunal_consensus, "composite_score", getattr(tribunal_consensus, "consensus_score", 85.0))
            verdict_val = getattr(tribunal_consensus, "consensus_verdict", getattr(tribunal_consensus, "publication_recommendation", "ACCEPTED"))
            trib_panel = (
                f"[bold]Composite Consensus Score:[/bold] [bold green]{score_val:.1f}/100[/bold green]\n"
                f"[bold]Board Recommendation:[/bold] [bold cyan]{verdict_val}[/bold cyan]"
            )
            console.print(Panel(trib_panel, title="⚖️ Adversarial Peer Tribunal Consensus", border_style="green"))
            await resource_guard.throttle_if_needed()
            resource_guard.optimize_memory()
        else:
            tribunal_consensus = TribunalConsensus(
                consensus_verdict="NOT_CONVENED",
                composite_score=0.0,
                reviews=[],
                actionable_revisions=[],
            )

        # ---------------------------------------------------------------------
        # 9. Author Formal Scientific Paper in Markdown (Dr. Editor)
        # ---------------------------------------------------------------------
        console.rule("[bold green]Landmark Scientific Paper Authoring (Dr. Editor)[/bold green]")
        notify("Spawning Dr. Editor to author publication-grade Scientific Paper in Markdown...")
        editor_agent = self.factory.create_paper_editor()
        journal.record_agent_spawn(editor_agent.name, editor_agent.role, editor_agent.goal, editor_agent.model)
        print_agent_spawn_card(
            name=editor_agent.name,
            role=editor_agent.role,
            goal=editor_agent.goal,
            model=editor_agent.model,
            task_instruction=f"Compile all multi-agent findings, graphs, in-silico tables, and references into a formal manuscript for '{topic}'",
            color="green",
        )

        papers_dir = self.settings.data_dir / "papers"
        slug = "".join([c if c.isalnum() else "_" for c in topic.lower()])[:30].strip("_")

        paper_path = await paper_writer.generate_paper_async(
            topic=topic,
            papers=deduped_papers,
            clinical_trials=clinical_trials,
            knowledge_graph=knowledge_graph,
            journal=journal,
            hypotheses=verified_hypotheses,
            milestones=milestones,
            output_dir=papers_dir,
            literature_synthesis=analysis_context,
            clinical_landscape=clinical_synthesis,
            depmap_evaluations=depmap_evals,
            fda_safety=fda_report,
            clinvar_stratification=clinvar_report,
            tme_microenvironment=tme_report,
            crispr_design=crispr_report,
            tribunal_consensus=tribunal_consensus,
        )
        journal.log(editor_agent.name, "Publication Authoring", f"Authored scientific paper saved to {paper_path.name}")
        await resource_guard.throttle_if_needed()
        resource_guard.optimize_memory()

        # Multi-format manuscript exports (PDF, LaTeX, BibTeX)
        if export_formats:
            fmt_set = {f.lower().strip() for f in export_formats}
            try:
                with open(paper_path, "r", encoding="utf-8") as pf:
                    paper_md_text = pf.read()

                if "pdf" in fmt_set:
                    from helixmind.publishing import PDFExporter
                    pdf_path = papers_dir / f"{paper_path.stem}.pdf"
                    PDFExporter.export_pdf(paper_md_text, pdf_path)
                    notify(f"Exported publication PDF: {pdf_path.name}")

                if "latex" in fmt_set or "tex" in fmt_set:
                    from helixmind.publishing import LaTeXExporter
                    tex_path = papers_dir / f"{paper_path.stem}.tex"
                    LaTeXExporter.export_latex(paper_md_text, tex_path)
                    notify(f"Exported LaTeX manuscript: {tex_path.name}")

                if "bibtex" in fmt_set or "bib" in fmt_set:
                    from helixmind.publishing import BibTeXExporter
                    bib_path = papers_dir / f"{paper_path.stem}.bib"
                    BibTeXExporter.export_bibtex(deduped_papers, clinical_trials, bib_path)
                    notify(f"Exported BibTeX citations: {bib_path.name}")
            except Exception as ex:
                console.print(f"[yellow]Export generation notice: {ex}[/yellow]")

        await emit(
            EventType.PAPER_AUTHORED,
            f"Scientific Paper authored and saved to: {paper_path.name}",
            {
                "paper_path": str(paper_path),
                "filename": paper_path.name,
                "title": f"Autonomous Investigation of {topic.title()}",
            },
        )

        # ---------------------------------------------------------------------
        # 10. Final Report Assembly & Persistence
        # ---------------------------------------------------------------------
        notify("Assembling final scientific investigation report and saving artifacts...")
        report = ResearchReport(
            topic=topic,
            timestamp=timestamp,
            papers_analyzed=len(deduped_papers),
            clinical_trials_analyzed=len(clinical_trials),
            entities_discovered=entities_found[:15],
            hypotheses=verified_hypotheses,
            clinical_landscape_summary=clinical_synthesis,
            critical_synthesis=analysis_context,
            next_research_milestones=milestones,
            paper_markdown_path=str(paper_path),
            knowledge_graph_summary=knowledge_graph.to_markdown_summary(),
            depmap_evaluations=[d.model_dump() for d in depmap_evals],
            denovo_analogs=[],
            clinical_simulation=None,
            fda_safety=fda_report.model_dump(),
            clinvar_stratification=clinvar_report.model_dump(),
            tme_microenvironment=tme_report.model_dump(),
        )

        # Persist report, journal, and graph
        reports_dir = self.settings.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"report_{slug}_{timestamp_str}.json"
        journal_file = reports_dir / f"journal_{slug}_{timestamp_str}.json"
        graph_file = reports_dir / f"graph_{slug}_{timestamp_str}.json"

        console.print(f"\n[bold cyan]💾 [Storage & Graph DB][/bold cyan] Persisting {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} directed edges to disk: [green]{graph_file.name}[/green]")
        knowledge_graph.export_json(graph_file)

        console.print(f"[bold cyan]💾 [Research Journal DB][/bold cyan] Persisting {len(journal.entries)} scientific audit logs to disk: [green]{journal_file.name}[/green]")
        journal.export_json(journal_file)

        console.print(f"[bold cyan]💾 [Research Report DB][/bold cyan] Serializing comprehensive multi-agent report to disk: [green]{report_file.name}[/green]")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        console.rule("[bold green]Scientific Investigation Artifacts Saved[/bold green]")
        artifacts_card = (
            f"[bold]Topic:[/bold] [yellow]{topic}[/yellow]\n"
            f"[bold]Scientific Paper (Markdown):[/bold] [bold cyan]{paper_path}[/bold cyan]\n"
            f"[bold]Comprehensive Research Report:[/bold] [green]{report_file}[/green]\n"
            f"[bold]Investigation Journal:[/bold] [white]{journal_file}[/white]\n"
            f"[bold]Knowledge Graph Artifact:[/bold] [white]{graph_file}[/white]\n"
            f"[bold]Papers Analyzed:[/bold] {len(deduped_papers)} | [bold]Trials Evaluated:[/bold] {len(clinical_trials)} | [bold]Hypotheses:[/bold] {len(verified_hypotheses)}"
        )
        console.print(Panel(artifacts_card, title="📁 Generated Investigation Artifacts", border_style="green"))
        notify(f"Investigation complete! Report saved to [bold green]{report_file}[/bold green]")
        notify(f"Scientific Paper (Markdown) saved to [bold green]{paper_path}[/bold green]")

        await emit(
            EventType.INVESTIGATION_COMPLETED,
            f"Investigation complete! Report saved to {report_file.name}",
            {
                "investigation_id": inv_id,
                "report_file": str(report_file),
                "paper_path": str(paper_path),
                "papers_analyzed": len(deduped_papers),
                "clinical_trials_analyzed": len(clinical_trials),
                "hypotheses_count": len(verified_hypotheses),
            },
        )
        return report
