"""Unit tests for In-Silico Chemoinformatics, Rigor Tribunal, Memory, and Publishing."""

from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from helixmind.graph.models import KnowledgeGraph
from helixmind.insilico import ChemoinformaticsEngine, DockingSimulator, StructureRetriever
from helixmind.journal import ResearchJournal
from helixmind.memory import MacroMemory
from helixmind.papers.indexer import SemanticIndex
from helixmind.papers.models import Paper
from helixmind.publishing import BibTeXExporter, LaTeXExporter, PDFExporter
from helixmind.rigor import FactChecker, PeerReviewTribunal


def test_chemoinformatics_lipinski():
    props = ChemoinformaticsEngine.analyze_molecule("tamoxifen")
    assert props is not None
    assert props.molecular_weight > 300
    assert props.lipinski_pass is True
    assert props.qed_score > 0.4
    assert props.chemical_formula == "C26H29NO"

    aspirin = ChemoinformaticsEngine.analyze_molecule("aspirin")
    assert aspirin is not None
    assert aspirin.lipinski_violations == 0
    assert aspirin.qed_score > 0.5


def test_docking_simulator_thermodynamics():
    pred = DockingSimulator.predict_docking_affinity("tamoxifen", "HER2")
    assert pred.binding_affinity_delta_g < -5.0
    assert pred.estimated_ki_micromolar > 0
    assert pred.pdb_id == "1N8Z"
    assert "High Affinity" in pred.druggability_verdict or "Moderate" in pred.druggability_verdict


def test_structure_retriever_mappings():
    assert StructureRetriever.resolve_pdb_id("HER2") == "1N8Z"
    assert StructureRetriever.resolve_pdb_id("PD-L1") == "5X8M"
    assert StructureRetriever.resolve_pdb_id("EGFR") == "1M17"


@pytest.mark.asyncio
async def test_fact_checker_grounding():
    journal = ResearchJournal(investigation_id="TEST-RIGOR", topic="Cancer")
    journal.record_extracted_evidence(
        paper_id="42669245",
        paper_title="Selenium nanocatalysts",
        selection_rationale="Tested in glioma",
        key_excerpt="Polyphenol-stabilized selenium nanoparticles cross blood brain barrier.",
        scientific_argumentation="Proves intracranial delivery.",
    )
    index = SemanticIndex()

    checker = FactChecker()
    with patch.object(index, "search_async", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        report = await checker.audit_hypothesis_async(
            hypothesis_title="Selenium nanoparticles cross blood brain barrier",
            mechanism_of_action="Targeted delivery across endothelial junctions",
            predictions=["Brain accumulation confirmed"],
            journal=journal,
            index=index,
        )
        assert report.total_claims_audited > 0
        assert report.overall_rigor_score > 0.0


@pytest.mark.asyncio
async def test_peer_review_tribunal():
    tribunal = PeerReviewTribunal()
    with patch("helixmind.agents.base.BaseAgent.execute_structured_task_async", new_callable=AsyncMock) as mock_exec:
        from helixmind.rigor.peer_review_tribunal import ReviewerScore, TribunalEvaluationOutput
        r = [
            ReviewerScore(reviewer_name="Dr. ADMET", domain="PK", score=85, verdict="ACCEPT", critique="Bioavailable", key_risk="Tox"),
            ReviewerScore(reviewer_name="Dr. Mechanist", domain="Bio", score=90, verdict="ACCEPT", critique="Plausible", key_risk="Escape"),
            ReviewerScore(reviewer_name="Dr. Biostat", domain="Stat", score=82, verdict="ACCEPT", critique="Solid", key_risk="Power"),
        ]
        mock_exec.return_value = TribunalEvaluationOutput(reviews=r, actionable_revisions=["Monitor Tox"])
        consensus = await tribunal.convene_tribunal_async(
            hypothesis_title="Targeting HER2 with Tamoxifen",
            mechanism="Selective receptor modulation",
            targets=["HER2"],
            predictions=["Cell death in vitro"],
        )
        assert len(consensus.reviews) == 3
        assert consensus.composite_score > 70
        assert "ACCEPT" in consensus.consensus_verdict or "ENDORSEMENT" in consensus.consensus_verdict


def test_macro_memory_cross_disease():
    mem = MacroMemory()
    g1 = KnowledgeGraph()
    g1.add_node("STAT3", "protein")
    g1.add_node("Glioblastoma", "disease")
    g1.add_edge("STAT3", "protein", "Glioblastoma", "disease", "upregulated_in")

    # Ingest Investigation 1
    mem.merge_investigation("INV-01", "Glioblastoma", g1)

    g2 = KnowledgeGraph()
    g2.add_node("STAT3", "protein")
    g2.add_node("Breast Cancer", "disease")
    g2.add_edge("STAT3", "protein", "Breast Cancer", "disease", "targets")

    # Ingest Investigation 2 (shares STAT3!)
    insights = mem.merge_investigation("INV-02", "Breast Cancer", g2)
    assert len(insights) > 0
    assert insights[0].shared_entity_name == "STAT3"
    assert "Glioblastoma" in insights[0].source_topic


def test_publishing_exporters(tmp_path: Path):
    paper = Paper(
        id="PMID:88888",
        title="Sample publication for export testing",
        abstract="Testing automated export to PDF, LaTeX, and BibTeX.",
        authors=["Valdez D", "HelixMind Team"],
        pub_year=2026,
        journal="Nature Bio-AI",
        doi="10.1038/s41586-026-0001",
    )

    # 1. Test BibTeX
    bib_path = tmp_path / "references.bib"
    BibTeXExporter.export_bibtex([paper], [], bib_path)
    assert bib_path.exists()
    bib_text = bib_path.read_text(encoding="utf-8")
    assert "@article{" in bib_text
    assert "Nature Bio-AI" in bib_text

    # 2. Test LaTeX
    md_text = "# Test Investigation\n\n## Abstract\nSample abstract text.\n\n## 1. Introduction\nSample body paragraph."
    tex_path = tmp_path / "article.tex"
    LaTeXExporter.export_latex(md_text, tex_path)
    assert tex_path.exists()
    tex_text = tex_path.read_text(encoding="utf-8")
    assert "\\documentclass" in tex_text
    assert "\\begin{abstract}" in tex_text

    # 3. Test PDF (ReportLab)
    pdf_path = tmp_path / "article.pdf"
    PDFExporter.export_pdf(md_text, pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000
