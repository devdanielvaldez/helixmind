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

from helixmind.agents.factory import AgentFactory, SearchStrategy
from helixmind.config import Settings, get_settings
from helixmind.hypotheses.models import (
    HypothesisCritique,
    HypothesisList,
    ResearchReport,
    ScientificHypothesis,
)
from helixmind.llm.ollama_client import OllamaClient
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
        model: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> ResearchReport:
        """Run an autonomous end-to-end investigation on a target condition or topic."""
        if model:
            self.factory = AgentFactory(ollama_client=self.ollama, model=model)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def notify(msg: str) -> None:
            if on_progress:
                on_progress(msg)
            console.print(f"[bold cyan]▶ {msg}[/bold cyan]")

        notify(f"Initiating autonomous investigation for: '{topic}'")

        # ---------------------------------------------------------------------
        # 1. Spawn Search Strategist & Formulate Queries
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Strategist to construct structured biomedical queries...")
        strategist = self.factory.create_search_strategist()

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
        except Exception as e:
            strategist.log(f"Structured strategy fallback: {e}", style="yellow")
            strategy = SearchStrategy(
                primary_condition=topic,
                pubmed_query=topic,
                clinical_trials_condition=topic,
                europe_pmc_query=topic,
                key_biological_targets=[],
            )

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
            try:
                # Fetch peer-reviewed and preprints
                regular = await self.epmc_client.search_async(strategy.europe_pmc_query, limit=max_papers // 2 or 2)
                preprints = await self.epmc_client.search_async(
                    strategy.europe_pmc_query,
                    limit=max_papers // 2 or 2,
                    preprints_only=True,
                )
                return regular + preprints
            except Exception as ex:
                console.print(f"[yellow]Europe PMC fetch notice: {ex}[/yellow]")
                return []

        async def fetch_trials():
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

        # Deduplicate papers by title or PMID
        seen_titles = set()
        deduped_papers: List[Paper] = []
        for p in pubmed_papers + epmc_papers:
            norm_title = p.title.strip().lower()
            if norm_title and norm_title not in seen_titles:
                seen_titles.add(norm_title)
                deduped_papers.append(p)

        notify(f"Retrieved {len(deduped_papers)} unique papers and {len(clinical_trials)} clinical trials.")

        # Mine biological annotations for top papers
        entities_found: List[str] = list(strategy.key_biological_targets)
        for p in deduped_papers[:4]:
            if p.pmid:
                ann_list = await self.epmc_client.fetch_annotations_async(p.pmid)
                p.entities = ann_list
                for ent in ann_list:
                    if ent.name not in entities_found and len(ent.name) > 2:
                        entities_found.append(ent.name)

        # ---------------------------------------------------------------------
        # 3. Vector RAG Indexing with nomic-embed-text
        # ---------------------------------------------------------------------
        notify("Vectorizing and indexing documents into fast Semantic RAG index...")
        indexed_count = await self.index.index_documents_async(
            papers=deduped_papers,
            studies=clinical_trials,
        )
        notify(f"Indexed {indexed_count} semantic chunks into in-memory vector space.")

        # ---------------------------------------------------------------------
        # 4. Literature Analysis Agent (Dr. Synapse)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Synapse to extract mechanisms and synthesize literature...")
        analyst = self.factory.create_literature_analyst()

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

        # ---------------------------------------------------------------------
        # 5. Clinical Trial Evaluator Agent (Dr. Trials)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Trials to evaluate human clinical trial pipelines...")
        trials_evaluator = self.factory.create_clinical_trial_evaluator()

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

        # ---------------------------------------------------------------------
        # 6. Progressive Part-by-Part Hypothesis Generation (Dr. Hypothesis)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Hypothesis with Reasoning model to formulate hypotheses progressively (part-by-part)...")
        generator = self.factory.create_hypothesis_generator()

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
                generator.log(f"Formulated {single_hypo.id}: '{single_hypo.title}'", style="green")
            except Exception as e:
                generator.log(f"Hypothesis {h_i} structured fallback: {e}", style="yellow")
                raw_text = await generator.execute_task_async(
                    task_instruction=prompt_single,
                    context=combined_evidence,
                    temperature=0.3,
                )
                raw_hypotheses.append(
                    ScientificHypothesis(
                        id=f"HYP-{h_i:02d}",
                        title=f"Targeted modulation of {entities_found[h_i-1] if len(entities_found)>=h_i else 'signaling pathway'} in {strategy.primary_condition}",
                        target_condition=strategy.primary_condition,
                        biological_targets=entities_found[h_i-1:h_i+2] if entities_found else ["Target Pathway"],
                        mechanism_of_action=raw_text[:400].strip(),
                        testable_predictions=["Differential expression in target phenotype", "Synergistic inhibition in-vitro"],
                        validation_strategy="In-vitro cell viability assay & CRISPR knockdown",
                        confidence_score=0.75,
                    )
                )

        # ---------------------------------------------------------------------
        # 7. Critical Peer-Review Agent (Dr. Skeptic / Devil's Advocate)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Skeptic (Devil's Advocate) to stress-test each hypothesis...")
        skeptic = self.factory.create_critical_reviewer()

        verified_hypotheses: List[ScientificHypothesis] = []
        for hypo in raw_hypotheses:
            critique_prompt = (
                f"Critique this hypothesis in 2 sentences:\n"
                f"Title: {hypo.title}\n"
                f"Targets: {', '.join(hypo.biological_targets)}\n"
                f"Mechanism: {hypo.mechanism_of_action[:200]}\n"
                "Issue a verdict and highlight potential off-target or biological risks."
            )

            try:
                critique: HypothesisCritique = await skeptic.execute_structured_task_async(
                    task_instruction=critique_prompt,
                    schema=HypothesisCritique,
                    context=rag_context[:400],
                    temperature=0.2,
                )
                hypo.critique_verdict = critique.verdict
                hypo.counter_arguments = critique.identified_risks + critique.unaddressed_confounders
                hypo.recommended_experiments = critique.falsification_tests
            except Exception:
                critique_text = await skeptic.execute_task_async(
                    task_instruction=critique_prompt,
                    context=rag_context[:400],
                )
                hypo.critique_verdict = "Plausible with Caveats"
                hypo.counter_arguments = [critique_text[:200]]

            verified_hypotheses.append(hypo)
            skeptic.log(f"Critique for {hypo.id} -> Verdict: [bold]{hypo.critique_verdict}[/bold]", style="magenta")

        # ---------------------------------------------------------------------
        # 8. Experimental Designer Agent (Dr. Protocol)
        # ---------------------------------------------------------------------
        notify("Spawning Dr. Protocol to synthesize detailed experimental roadmap...")
        designer = self.factory.create_experiment_designer()

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

        # ---------------------------------------------------------------------
        # 9. Final Report Assembly & Persistence
        # ---------------------------------------------------------------------
        notify("Assembling final scientific investigation report...")
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
        )

        # Persist report
        reports_dir = self.settings.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        slug = "".join([c if c.isalnum() else "_" for c in topic.lower()])[:30]
        report_file = reports_dir / f"report_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        notify(f"Investigation complete! Report saved to [bold green]{report_file}[/bold green]")
        return report
