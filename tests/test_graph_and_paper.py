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
    assert "## 8. References" in content
