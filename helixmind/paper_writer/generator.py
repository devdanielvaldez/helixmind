"""Scientific Paper Generator: compiles multi-agent investigation findings into a formal Markdown paper."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional
from rich.console import Console

from helixmind.agents.base import BaseAgent
from helixmind.graph.models import KnowledgeGraph
from helixmind.hypotheses.models import ScientificHypothesis
from helixmind.journal.journal import ResearchJournal
from helixmind.llm.ollama_client import OllamaClient
from helixmind.papers.models import ClinicalStudy, Paper

console = Console()


class ScientificPaperGenerator:
    """Generates a formal, publication-ready scientific paper in GitHub Markdown."""

    def __init__(self, ollama_client: Optional[OllamaClient] = None, model: Optional[str] = None):
        self.client = ollama_client or OllamaClient()
        self.model = model

    def create_editor_agent(self) -> BaseAgent:
        """Spawn the Chief Scientific Editor agent."""
        system_prompt = (
            "You are Dr. Editor, Editor-in-Chief of a high-impact scientific biomedical journal.\n"
            "Your task is to synthesize findings from a consortium of specialized autonomous agents\n"
            "(literature reviewers, bioinformaticians, clinical trialists, hypothesis formulators)\n"
            "into a formal, academic scientific paper in Markdown. Maintain academic rigor, precise terminology, "
            "and evidence-based reasoning."
        )
        return BaseAgent(
            name="Dr. Editor",
            role="Editor-in-Chief & Lead Author",
            goal="Synthesize multi-agent findings into a formal scientific publication in Markdown",
            system_prompt=system_prompt,
            model=self.model,
            temperature=0.3,
            ollama_client=self.client,
        )

    async def generate_paper_async(
        self,
        topic: str,
        papers: List[Paper],
        clinical_trials: List[ClinicalStudy],
        knowledge_graph: KnowledgeGraph,
        journal: ResearchJournal,
        hypotheses: List[ScientificHypothesis],
        milestones: List[str],
        output_dir: Optional[Path] = None,
        literature_synthesis: str = "",
        clinical_landscape: str = "",
    ) -> Path:
        """Progressively author and assemble the full scientific paper in Markdown."""
        editor = self.create_editor_agent()
        editor.log("Commencing publication-grade paper synthesis...", style="bold green")

        date_str = datetime.now().strftime("%B %d, %Y")
        slug = "".join([c if c.isalnum() else "_" for c in topic.lower()])[:30].strip("_")

        # ---------------------------------------------------------------------
        # 1. Author Abstract & Introduction (Dr. Editor)
        # ---------------------------------------------------------------------
        intro_prompt = (
            f"Topic: '{topic}'.\n"
            f"Analyzed {len(papers)} publications/preprints and {len(clinical_trials)} clinical trials.\n"
            f"Key targets identified: {', '.join(list(knowledge_graph.nodes.keys())[:8])}.\n"
            "Write a formal structured academic Abstract (Background, Methods, Findings, Conclusions) and "
            "a concise Introduction on clinical challenges and knowledge gaps (2 paragraphs)."
        )
        intro_combined = await editor.execute_task_async(intro_prompt)
        
        # Split or format abstract and intro
        if "introduction" in intro_combined.lower():
            parts = re.split(r"(?i)#*\s*introduction", intro_combined, maxsplit=1)
            abstract_text = parts[0].replace("#", "").replace("Abstract:", "").strip()
            intro_text = parts[1].strip() if len(parts) > 1 else intro_combined
        else:
            abstract_text = intro_combined[:600].strip()
            intro_text = intro_combined[600:].strip() or intro_combined

        # ---------------------------------------------------------------------
        # 2. Systems Biology & Clinical Landscape Analysis
        # ---------------------------------------------------------------------
        graph_mermaid = knowledge_graph.to_mermaid(max_edges=35)
        graph_table = knowledge_graph.to_epistemic_matrix_markdown()
        graph_discussion = literature_synthesis or "Cross-examination of literature highlights critical hub nodes and feedback loops governing target activation."
        clinical_text = clinical_landscape or "Clinical evaluation demonstrates a scarcity of translational trials, emphasizing the urgency of validating preclinical hypotheses."

        # ---------------------------------------------------------------------
        # 5. Compile Defined Terminology Table
        # ---------------------------------------------------------------------
        glossary_rows = []
        if journal.glossary:
            for term_obj in sorted(journal.glossary.values(), key=lambda t: t.term):
                glossary_rows.append(f"| **{term_obj.term}** | `{term_obj.category}` | {term_obj.definition} |")
        else:
            for node in list(knowledge_graph.nodes.values())[:10]:
                glossary_rows.append(f"| **{node.name}** | `{node.category}` | Key biological entity identified in evidence base |")

        glossary_table = (
            "| Term / Biomarker | Category | Mechanistic Definition |\n"
            "|---|---|---|\n" + "\n".join(glossary_rows)
        )

        # ---------------------------------------------------------------------
        # 6. Format Hypotheses & Peer-Review Critiques
        # ---------------------------------------------------------------------
        hypotheses_markdown = []
        for i, hyp in enumerate(hypotheses, 1):
            verdict_badge = hyp.critique_verdict or "Plausible with Caveats"
            caveats_list = "\n".join([f"  - {c}" for c in hyp.counter_arguments]) if hyp.counter_arguments else "  - Standard pharmacokinetic and off-target screening required."
            predictions_list = "\n".join([f"  1. {p}" for p in hyp.testable_predictions]) if hyp.testable_predictions else "  1. Downregulation of target marker upon intervention."

            hyp_block = (
                f"### Hypothesis {i}: {hyp.title}\n\n"
                f"- **Hypothesis ID:** `{hyp.id}`\n"
                f"- **Biological Targets:** {', '.join(hyp.biological_targets)}\n"
                f"- **Confidence Score:** {hyp.confidence_score * 100:.0f}%\n"
                f"- **Peer-Review Verdict (Devil's Advocate):** **`{verdict_badge}`**\n\n"
                f"#### Proposed Mechanism of Action (MoA)\n{hyp.mechanism_of_action}\n\n"
                f"#### Empirical Falsifiable Predictions\n{predictions_list}\n\n"
                f"#### Critical Review & Potential Risks\n{caveats_list}\n\n"
                f"#### Validation Strategy\n{hyp.validation_strategy}\n"
            )
            hypotheses_markdown.append(hyp_block)

        # ---------------------------------------------------------------------
        # 7. Format Experimental Roadmap
        # ---------------------------------------------------------------------
        roadmap_items = "\n".join([f"- **{m}**" if not m.startswith("-") else m for m in milestones]) or "- Comprehensive in-vitro cell assay and in-vivo preclinical modeling."

        # ---------------------------------------------------------------------
        # 8. References / Bibliography
        # ---------------------------------------------------------------------
        ref_lines = []
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
            ref_entry = f"{i}. {authors_str} ({p.pub_year or 'N/A'}). **{p.title}**."
            if p.journal:
                ref_entry += f" *{p.journal}*."
            if p.pmid:
                ref_entry += f" [PubMed PMID:{p.pmid}](https://pubmed.ncbi.nlm.nih.gov/{p.pmid}/)."
            if p.pmcid:
                ref_entry += f" [PMC {p.pmcid}](https://www.ncbi.nlm.nih.gov/pmc/articles/{p.pmcid}/)."
            if p.doi:
                ref_entry += f" DOI: {p.doi}"
            ref_lines.append(ref_entry)

        for i, t in enumerate(clinical_trials, len(papers) + 1):
            ref_lines.append(
                f"{i}. ClinicalTrials.gov ({t.status}). **{t.title}**. Identifier: [{t.nct_id}](https://clinicaltrials.gov/study/{t.nct_id})."
            )

        references_block = "\n".join(ref_lines) if ref_lines else "1. Internal Literature and Systems Biology Index, HelixMind Platform."

        # ---------------------------------------------------------------------
        # 9. Assemble Complete Markdown Paper
        # ---------------------------------------------------------------------
        full_paper = f"""# Autonomous Investigation of {topic.title()}: Mechanistic Network Interactions, Emergent Hypotheses, and Experimental Validation Framework

**Authors:** HelixMind Autonomous Multi-Agent Consortium  
*(Dr. Strategist, Dr. Synapse, Dr. Trials, Dr. Hypothesis, Dr. Skeptic, Dr. Graph, Dr. Protocol, and Dr. Editor)*  
**Date:** {date_str}  
**Platform Version:** HelixMind v0.1.0  
**Repository ID:** `{journal.investigation_id}`

---

## Abstract
{abstract_text.strip()}

**Keywords:** {topic}, systems biology, targeted therapy, knowledge graph, autonomous research, precision oncology

---

## 1. Defined Scientific Terminology & Conceptual Ontology
The following biological targets, compounds, and disease phenotypes were categorized and integrated across literature and trial databases:

{glossary_table}

---

## 2. Introduction & Disease Landscape
{intro_text.strip()}

---

## 3. Systematic Evidence Extraction, Excerpt Analysis & Argumentation Matrix
The autonomous multi-agent consortium conducted systematic evidence extraction from primary literature and preprints. For each selected publication, agents documented the selection rationale, extracted key empirical fragments, and provided mechanistic scientific argumentation:

{journal.to_evidence_markdown()}

---

## 4. Systems Biology & Epistemic Knowledge Graph Network
Through cross-database literature and preprint mining, the platform constructed an integrated epistemic knowledge network connecting empirical evidence, molecular concepts, testable hypotheses, critical doubts, and experimental conclusions.

### Epistemic Knowledge Network Flowchart
{graph_mermaid}

### Epistemic Connections Matrix
{graph_table}

### Mechanistic Network Discussion
{graph_discussion.strip()}

---

## 5. Translational Landscape & Clinical Trials Evidence
{clinical_text.strip()}

---

## 6. Formulated Scientific Hypotheses & Mechanistic Models
Grounded in the synthesized literature and knowledge graph topology, the autonomous reasoning agent formulated the following testable scientific hypotheses, subjected to adversarial peer review:

{"".join(hypotheses_markdown)}

---

## 7. Actionable Experimental Validation Roadmap
To empirically validate or falsify the surviving hypotheses, the autonomous experimental architect outlines the following sequential validation phases:

{roadmap_items}

---

## 8. Autonomous Discovery Audit & Methodology
This investigation was autonomously orchestrated by specialized multi-agents executing iterative literature ingestion (PubMed, Europe PMC, Preprints), clinical trials mining (ClinicalTrials.gov), vector RAG semantic search (`nomic-embed-text`), and adversarial peer-review stress-testing.

- **Total Publications & Preprints Analyzed:** {len(papers)}
- **Clinical Trials Evaluated:** {len(clinical_trials)}
- **Extracted Evidence Items:** {len(journal.evidence_dossier)}
- **Knowledge Graph Nodes:** {len(knowledge_graph.nodes)}
- **Knowledge Graph Edges:** {len(knowledge_graph.edges)}
- **Agents Deployed:** {len(journal.agents_roster)} ({', '.join([a.name for a in journal.agents_roster])})

---

## 9. References
{references_block}
"""

        # Save to disk
        target_dir = output_dir or Path("data/papers")
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        paper_path = target_dir / f"paper_{slug}_{timestamp_str}.md"

        with open(paper_path, "w", encoding="utf-8") as f:
            f.write(full_paper)

        editor.log(f"Scientific Paper authored and saved to: {paper_path}", style="bold green")
        return paper_path
