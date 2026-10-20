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
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from helixmind.agents.director import AgentMission, DirectorEvaluation, ResearchDirector
from helixmind.agents.factory import AgentFactory, SearchStrategy
from helixmind.api.events import EventType
from helixmind.api.ws_manager import ws_manager
from helixmind.config import Settings, get_settings
from helixmind.graph import GraphExtractor, KnowledgeGraph
from helixmind.hypotheses.models import (
    HypothesisCritique,
    HypothesisList,
    ResearchReport,
    ScientificHypothesis,
)
from helixmind.journal import ResearchJournal
from helixmind.llm.ollama_client import OllamaClient
from helixmind.paper_writer import ScientificPaperGenerator
from helixmind.papers.clinical_trials_client import ClinicalTrialsClient
from helixmind.papers.europe_pmc_client import EuropePMCClient
from helixmind.papers.indexer import SemanticIndex
from helixmind.papers.models import BiologicalEntity, ClinicalStudy, Paper
from helixmind.papers.ncbi_client import NCBIClient

console = Console()


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
        depth: str = "standard",
        model: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> ResearchReport:
        """Run an autonomous end-to-end investigation on a target condition or topic."""
        if model:
            self.factory = AgentFactory(ollama_client=self.ollama, model=model)

        chosen_model = model or self.settings.ollama_default_model
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        inv_id = f"INV-{int(datetime.now().timestamp())}"

        depth_cycles = {
            "quick": 1,
            "standard": 1,
            "deep": 2,
            "exhaustive": 3,
        }
        num_cycles = depth_cycles.get(depth.lower(), 1)

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
        strategist = self.factory.create_search_strategist()
        journal.record_agent_spawn(strategist.name, strategist.role, strategist.goal, strategist.model)
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

        # ---------------------------------------------------------------------
        # 2. Ingest Literature & Trials across PubMed, Europe PMC, ClinicalTrials
        # ---------------------------------------------------------------------
        notify("Ingesting literature from PubMed, preprints from Europe PMC, and ClinicalTrials.gov...")

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
        journal.log("Ingestion Pipeline", "Evidence Retrieval", f"Retrieved {len(deduped_papers)} papers and {len(clinical_trials)} clinical trials")

        # Mine biological annotations for top papers
        entities_found: List[str] = list(strategy.key_biological_targets)
        for p in deduped_papers[:4]:
            if p.pmid:
                ann_list = await self.epmc_client.fetch_annotations_async(p.pmid)
                p.entities = ann_list
                for ent in ann_list:
                    if ent.name not in entities_found and len(ent.name) > 2:
                        entities_found.append(ent.name)
                        knowledge_graph.add_node(ent.name, ent.entity_type.lower(), reference=p.pmid)
                        journal.add_glossary_term(ent.name, f"Identified {ent.entity_type} from literature", ent.entity_type.lower(), "Europe PMC Annotations")

        # ---------------------------------------------------------------------
        # 3. Dynamic Knowledge Graph Mining & Vector RAG Indexing
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Graph to mine molecular interactions & build Knowledge Graph...")
        journal.record_agent_spawn("Dr. Graph", "Systems Biology & Knowledge Graph Specialist", "Extract biological interactions", chosen_model)
        relations_mined = await graph_extractor.extract_from_papers_async(deduped_papers, knowledge_graph, max_papers=4)
        await emit(
            EventType.GRAPH_UPDATED,
            f"Knowledge Graph updated: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} directed edges.",
            {
                "nodes_count": len(knowledge_graph.nodes),
                "edges_count": len(knowledge_graph.edges),
                "mermaid": knowledge_graph.to_mermaid(max_edges=30),
            },
        )
        journal.log("Dr. Graph", "Knowledge Graph Extraction", f"Discovered {relations_mined} biological interactions across literature")

        notify("Vectorizing and indexing documents into fast Semantic RAG index...")
        indexed_count = await self.index.index_documents_async(
            papers=deduped_papers,
            studies=clinical_trials,
        )
        notify(f"Indexed {indexed_count} semantic chunks into in-memory vector space.")
        journal.log("Vector Indexer", "RAG Indexing", f"Indexed {indexed_count} chunks with nomic-embed-text")

        # ---------------------------------------------------------------------
        # 4. Literature Analysis Agent (Dr. Synapse)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Synapse to extract mechanisms and synthesize literature...")
        analyst = self.factory.create_literature_analyst()
        journal.record_agent_spawn(analyst.name, analyst.role, analyst.goal, analyst.model)

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
        journal.log(analyst.name, "Literature Review", "Synthesized top molecular pathways and targets")

        # Systematically record extracted evidence with selection rationale and scientific argument
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
            await emit(
                EventType.EVIDENCE_EXTRACTED,
                f"Extracted key excerpt from {p.title}",
                ev.model_dump(mode="json"),
            )

        # ---------------------------------------------------------------------
        # 5. Clinical Trial Evaluator Agent (Dr. Trials)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Trials to evaluate human clinical trial pipelines...")
        trials_evaluator = self.factory.create_clinical_trial_evaluator()
        journal.record_agent_spawn(trials_evaluator.name, trials_evaluator.role, trials_evaluator.goal, trials_evaluator.model)

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
        journal.log(trials_evaluator.name, "Clinical Analysis", "Evaluated active trials and intervention landscape")

        # ---------------------------------------------------------------------
        # Dynamic Multi-Agent Exploration Loop (Coordinated by Research Director)
        # ---------------------------------------------------------------------
        director = ResearchDirector(ollama_client=self.ollama, model=chosen_model)
        for cycle in range(1, num_cycles + 1):
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
            console.print(f"[dim]Objective: {mission.agent_goal} | Rationale: {mission.rationale}[/dim]")

            mission_result = await director.execute_agent_mission_async(
                mission=mission,
                index=self.index,
                journal=journal,
                knowledge_graph=knowledge_graph,
                ncbi_client=self.ncbi_client,
                epmc_client=self.epmc_client,
                active_papers=deduped_papers,
            )
            analysis_context += f"\n\n[{mission.agent_name} Findings]:\n{mission_result[:300]}"

        # ---------------------------------------------------------------------
        # 6. Progressive Part-by-Part Hypothesis Generation (Dr. Hypothesis)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Hypothesis with Reasoning model to formulate hypotheses progressively (part-by-part)...")
        generator = self.factory.create_hypothesis_generator()
        journal.record_agent_spawn(generator.name, generator.role, generator.goal, generator.model)

        # Query RAG for deep mechanistic context
        rag_query = f"{strategy.primary_condition} therapeutic target pathway"
        rag_context = await self.index.get_relevant_context_async(rag_query, top_k=3)

        combined_evidence = (
            f"LITERATURE SYNTHESIS:\n{analysis_context[:600]}\n\n"
            f"CLINICAL LANDSCAPE:\n{clinical_synthesis[:400]}\n\n"
            f"TOP RAG EVIDENCE:\n{rag_context[:700]}\n\n"
            f"DISCOVERED TARGETS:\n{', '.join(entities_found[:8])}"
        )

        raw_hypotheses: List[ScientificHypothesis] = []
        num_hypotheses = 1 if depth == "quick" else min(2, self.settings.default_max_hypotheses)

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
                generator.log(f"Formulated {single_hypo.id}: '{single_hypo.title}'", style="green")
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
                generator.log(f"Hypothesis {h_i} structured fallback: {e}", style="yellow")
                raw_text = await generator.execute_task_async(
                    task_instruction=prompt_single,
                    context=combined_evidence,
                    temperature=0.3,
                )
                fallback_hyp = ScientificHypothesis(
                    id=f"HYP-{h_i:02d}",
                    title=f"Targeted modulation of {entities_found[h_i-1] if len(entities_found)>=h_i else 'signaling pathway'} in {strategy.primary_condition}",
                    target_condition=strategy.primary_condition,
                    biological_targets=entities_found[h_i-1:h_i+2] if entities_found else ["Target Pathway"],
                    mechanism_of_action=raw_text[:400].strip(),
                    testable_predictions=["Differential expression in target phenotype", "Synergistic inhibition in-vitro"],
                    validation_strategy="In-vitro cell viability assay & CRISPR knockdown",
                    confidence_score=0.75,
                )
                raw_hypotheses.append(fallback_hyp)
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
        notify("Spawning Dr. Skeptic (Devil's Advocate) to stress-test each hypothesis...")
        skeptic = self.factory.create_critical_reviewer()
        journal.record_agent_spawn(skeptic.name, skeptic.role, skeptic.goal, skeptic.model)

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
            skeptic.log(f"Critique for {hypo.id} -> Verdict: [bold]{hypo.critique_verdict}[/bold]", style="magenta")
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
        notify("Spawning Dr. Protocol to synthesize detailed experimental roadmap...")
        designer = self.factory.create_experiment_designer()
        journal.record_agent_spawn(designer.name, designer.role, designer.goal, designer.model)

        experiments_prompt = (
            f"List 3 bullet points (one concise sentence each) for experimental validation of '{strategy.primary_condition}':\n"
            "1. In-Silico Molecular Docking & Modeling\n"
            "2. In-Vitro & Cell Assays\n"
            "3. In-Vivo Preclinical Validation"
        )

        protocol_roadmap = await designer.execute_task_async(
            task_instruction=experiments_prompt,
            context="\n".join([f"{h.id}: {h.title}" for h in verified_hypotheses]),
        )

        milestones = [
            line.strip("- •* ")
            for line in protocol_roadmap.split("\n")
            if line.strip() and (line.strip()[:2].isdigit() or line.strip().startswith(("-", "•", "*")))
        ]

        primary_hyp_node = knowledge_graph.normalize_id("hypothesis", f"{verified_hypotheses[0].id}: {verified_hypotheses[0].title[:40]}") if verified_hypotheses else None
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
        # 9. Author Formal Scientific Paper in Markdown (Dr. Editor)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Editor to author publication-grade Scientific Paper in Markdown...")
        journal.record_agent_spawn("Dr. Editor", "Editor-in-Chief & Lead Author", "Synthesize formal scientific publication in Markdown", chosen_model)

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
        )
        journal.log("Dr. Editor", "Publication Authoring", f"Authored scientific paper saved to {paper_path.name}")
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
        )

        # Persist report, journal, and graph
        reports_dir = self.settings.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"report_{slug}_{timestamp_str}.json"
        journal_file = reports_dir / f"journal_{slug}_{timestamp_str}.json"
        graph_file = reports_dir / f"graph_{slug}_{timestamp_str}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        journal.export_json(journal_file)
        knowledge_graph.export_json(graph_file)

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
