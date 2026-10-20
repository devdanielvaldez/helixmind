"""Unit tests for PDF parsing, CRISPR/primer designer, molecular/DNA visualizer, and copilot."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from helixmind.copilot import ResearchCopilot
from helixmind.insilico import BioVisualizer, CRISPRDesigner
from helixmind.journal import ResearchJournal
from helixmind.papers import PDFParser, Paper


def test_crispr_designer_sgrnas_and_primers():
    designer = CRISPRDesigner()
    # 350-bp synthetic DNA containing multiple SpCas9 NGG PAM sites
    test_dna = (
        "ATGGAGAGCGACGAGAGCGGCCTGCCCGCCATGGAGATCGAGTGCCGCATCACCGGCACCCTGAACGGCGTGGAGTTCGAGCTGGTGGGCGGCGGAGAGGGCACCCCCGAGCAGGGCCGCATGACCAACAAGATGAAGAGCACCAAAGGCGCCCTGACCTTCAGCCCCTACCTGCTGAGCCACGTGATGGGCTACGGCTTCTACCACTTCGGCACCTACCCCAGCGGCTACGAGAACCCCTTCCTGCACGCCATCAACAACGGCGGCTACACCAACACCCGCATCGAGAAGTACGAGGACGGCGGCGTGCTGCACGTGAGCTTCAGCTACCGCTACGAGGCCGGCCGCGTGATCGGCGACTTCAAG"
    )

    guides = designer.design_sgrnas(test_dna, max_candidates=3)
    assert len(guides) > 0
    g1 = guides[0]
    assert len(g1.sequence_20nt) == 20
    assert g1.pam_motif.endswith("GG")
    assert 30.0 <= g1.gc_percent <= 75.0
    assert "CACC" in g1.oligo_forward_top
    assert "AAAC" in g1.oligo_reverse_bottom

    primers = designer.design_validation_primers(test_dna)
    assert primers is not None
    assert primers.forward_primer.startswith("5'-")
    assert primers.reverse_primer.startswith("5'-")
    assert primers.forward_tm > 50.0
    assert primers.amplicon_length_bp > 100


def test_bio_visualizer_molecule_and_dna(tmp_path: Path):
    # 1. 2D Molecule SVG
    mol_svg = BioVisualizer.render_molecule_svg("tamoxifen")
    assert mol_svg is not None
    assert "<svg" in mol_svg
    assert "</svg>" in mol_svg

    # 2. DNA Construct SVG
    dna_svg = BioVisualizer.render_dna_construct_svg("pCas9-sgRNA-Pur")
    assert "<svg" in dna_svg
    assert "GENETIC CONSTRUCT:" in dna_svg

    # 3. Figure saving
    fig_path = BioVisualizer.save_figure(dna_svg, "test_dna_figure")
    assert fig_path.exists()
    assert fig_path.stat().st_size > 100


def test_pdf_parser_chunking():
    parser = PDFParser()
    paper = Paper(
        id="LOCAL_TEST",
        source="local_pdf",
        title="Laboratory Preclinical Study on Nanoparticles",
        abstract="In this manuscript we present evidence for liposomal drug delivery.",
        full_text="1. Introduction\n\nLiposomes enhance delivery.\n\n2. Results\n\nTumor volume decreased significantly.",
    )

    chunks = parser.extract_chunks(paper)
    assert len(chunks) >= 2
    assert chunks[0].doc_id == "LOCAL_TEST"
    assert "Title: Laboratory Preclinical Study" in chunks[0].content


@pytest.mark.asyncio
async def test_research_copilot_steering():
    copilot = ResearchCopilot()
    journal = ResearchJournal(investigation_id="TEST-COPILOT", topic="Glioblastoma")
    journal.record_extracted_evidence(
        paper_id="42669245",
        paper_title="Selenium nanocatalysts",
        selection_rationale="Nose-to-brain delivery",
        key_excerpt="Selenium NPs cross blood brain barrier",
        scientific_argumentation="Solves delivery obstacle",
    )

    with patch("helixmind.agents.base.BaseAgent.execute_task_async", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = "Based on PMID: 42669245, selenium nanocatalysts enable nose-to-brain delivery. I have designed the CRISPR primers for CD274."
        response = await copilot.chat_async(
            investigation_id="TEST-COPILOT",
            user_message="Diseña primers de CRISPR para CD274 y cita el paper",
            topic="Glioblastoma",
            journal=journal,
        )

        assert "42669245" in response.reply
        assert len(response.actions_taken) > 0
        assert any("CRISPR" in a for a in response.actions_taken)
