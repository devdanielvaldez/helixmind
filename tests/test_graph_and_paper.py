"""Unit tests for Knowledge Graph, Research Journal, and Scientific Paper Generator."""

from unittest.mock import AsyncMock, patch
from pathlib import Path
import pytest
from helixmind.graph import GraphExtractor, KnowledgeGraph
from helixmind.hypotheses.models import ScientificHypothesis
from helixmind.journal import ResearchJournal
from helixmind.paper_writer import ScientificPaperGenerator
from helixmind.papers.models import Paper


def test_knowledge_graph_construction():
    graph = KnowledgeGraph()
    n1 = graph.add_node("HER2", "protein", "Receptor tyrosine kinase")
    n2 = graph.add_node("Trastuzumab", "drug", "Monoclonal antibody")

    edge = graph.add_edge(
        source_name="Trastuzumab",
        source_category="drug",
        target_name="HER2",
        target_category="protein",
        relation="inhibits",
        evidence="Binding leads to internalization and degradation",
        citation="PMID:12345",
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert edge.relation == "inhibits"

    # Test Mermaid generation
    mermaid = graph.to_mermaid()
    assert "flowchart TD" in mermaid
    assert "HER2" in mermaid
    assert "Trastuzumab" in mermaid

    # Test Markdown summary
    summary = graph.to_markdown_summary()
    assert "| **Trastuzumab** |" in summary
    assert "| **HER2** |" in summary


def test_epistemic_knowledge_graph_connections():
    graph = KnowledgeGraph()

    # 1. Add Evidence
    ev_node = graph.add_epistemic_evidence(
        paper_id="42669245",
        paper_title="Selenium nanocatalysts",
        excerpt="Nanocatalysts cross the blood brain barrier",
        rationale="Overcomes delivery challenge in glioblastoma",
    )
    assert "evidence:evidence_42669245" in graph.nodes

    # 2. Add Hypothesis wired to targets and evidence
    hyp_node = graph.add_epistemic_hypothesis(
        hypo_id="HYP-01",
        title="Selenium dual therapy",
        mechanism="Redox nanocatalysts trigger macrophage reprogramming",
        biological_targets=["PD-L1", "Selenium_NPs"],
        supporting_paper_ids=["42669245"],
    )
    assert hyp_node.category == "hypothesis"
    assert any(e.relation == "supports" for e in graph.edges)
    assert any(e.relation == "targets" for e in graph.edges)

    # 3. Add Doubt / Risk
    doubt_node = graph.add_epistemic_doubt(
        doubt_statement="Off-target toxicity in healthy astrocytes",
        challenged_hypothesis_node_id=hyp_node.id,
    )
    assert doubt_node.category == "doubt"
    assert any(e.relation == "challenges" for e in graph.edges)

    # 4. Add Conclusion
    conc_node = graph.add_epistemic_conclusion(
        conclusion_title="In-vitro cell viability milestone",
        statement="MTT assay confirms selective cytotoxicity",
        derived_from_hypothesis_node_id=hyp_node.id,
    )
    assert conc_node.category == "conclusion"
    assert any(e.relation == "derives_from" for e in graph.edges)

    # 5. Check Epistemic Matrix
    matrix_md = graph.to_epistemic_matrix_markdown()
    assert "CHALLENGES" in matrix_md
    assert "SUPPORTS" in matrix_md
    assert "DERIVES_FROM" in matrix_md

    # 6. Check Mermaid
    mermaid = graph.to_mermaid()
    assert '{{"HYPOTHESIS:' in mermaid
    assert '[/"DOUBT / RISK:' in mermaid
    assert "-.-|challenges|-.-" in mermaid


def test_research_journal_and_glossary():
    journal = ResearchJournal(investigation_id="INV-TEST-01", topic="Breast Cancer")

    journal.record_agent_spawn("Dr. Strategist", "Architect", "Create queries", "llama3.2:1b")
    journal.log("Dr. Strategist", "Planning", "Formulated initial MeSH queries")

    term = journal.add_glossary_term(
        term="PD-L1",
        definition="Programmed death-ligand 1 immune checkpoint protein",
        category="immune_checkpoint",
    )

    assert len(journal.agents_roster) == 1
    assert len(journal.entries) == 1
    assert "pd-l1" in journal.glossary
    assert term.term == "PD-L1"

    md = journal.to_markdown()
    assert "# Research Journal: Breast Cancer" in md
    assert "PD-L1" in md
    assert "Dr. Strategist" in md


def test_extracted_evidence_and_argumentation():
    journal = ResearchJournal(investigation_id="INV-TEST-EV", topic="Immunotherapy")

    ev = journal.record_extracted_evidence(
        paper_id="PMID:11111",
        paper_title="Targeting PD-1 in glioblastoma",
        selection_rationale="Provides evidence for T-cell exhaustion reversal in GBM",
        key_excerpt="Nivolumab enhanced CD8+ infiltration into tumor tissue by 45%.",
        scientific_argumentation="Proves that PD-1 blockade overcomes local immunosuppression.",
        agent_name="Dr. Synapse",
    )

    assert len(journal.evidence_dossier) == 1
    assert ev.paper_id == "PMID:11111"
    assert "CD8+ infiltration" in ev.key_excerpt

    ev_md = journal.to_evidence_markdown()
    assert "Evidence Item 1: Targeting PD-1 in glioblastoma" in ev_md
    assert "Selection Rationale" in ev_md
    assert "Scientific Argumentation" in ev_md


@pytest.mark.asyncio
async def test_paper_generator_structure(tmp_path: Path):
    graph = KnowledgeGraph()
    graph.add_edge("PD-1", "protein", "T-cells", "process", "inhibits", "Immune exhaustion")

    journal = ResearchJournal(investigation_id="INV-001", topic="Melanoma Immunotherapy")
    journal.add_glossary_term("Nivolumab", "Anti-PD-1 monoclonal antibody", "drug")

    paper = Paper(
        id="PMID:99999",
        title="Nivolumab efficacy in advanced melanoma",
        abstract="Nivolumab produced sustained tumor regressions in patients with metastatic melanoma.",
        authors=["Topalian SL", "Hodi FS"],
        pub_year=2024,
    )

    hypothesis = ScientificHypothesis(
        id="HYP-01",
        title="Dual checkpoint inhibition overcomes innate resistance",
        target_condition="Melanoma",
        biological_targets=["PD-1", "LAG-3"],
        mechanism_of_action="Concurrent blockade restores cytotoxic T-cell function.",
        testable_predictions=["Increased IFN-gamma secretion in-vitro"],
        validation_strategy="Flow cytometry and tumor infiltrate analysis",
        confidence_score=0.85,
    )

    writer = ScientificPaperGenerator(model="llama3.2:1b")
    with patch("helixmind.agents.base.BaseAgent.execute_task_async", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = "Synthesized academic narrative analyzing cellular mechanisms and therapeutic endpoints."
        paper_path = await writer.generate_paper_async(
            topic="Melanoma Immunotherapy",
            papers=[paper],
            clinical_trials=[],
            knowledge_graph=graph,
            journal=journal,
            hypotheses=[hypothesis],
            milestones=["1. In-silico docking", "2. In-vitro assay", "3. In-vivo validation"],
            output_dir=tmp_path,
        )

    assert paper_path.exists()
    content = paper_path.read_text(encoding="utf-8")
    assert "# Autonomous Investigation of Melanoma Immunotherapy" in content
    assert "## Abstract" in content
    assert "flowchart TD" in content
    assert "Hypothesis 1: Dual checkpoint inhibition overcomes innate resistance" in content
    assert "## 3. Systematic Evidence Extraction" in content
    assert "## 4. Systems Biology & Epistemic Knowledge Graph Network" in content
    assert "## 6. DepMap Functional Genomics" in content
    assert "## 7. In-Silico De Novo Lead Optimization" in content
    assert "## 8. Virtual Patient Clinical Trial Simulation" in content
    assert "## 12. References" in content
