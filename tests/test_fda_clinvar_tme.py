"""Unit tests for openFDA FAERS pharmacovigilance, NCBI ClinVar stratification, and Single-Cell TME."""

from unittest.mock import AsyncMock, patch
import pytest
from helixmind.insilico import (
    ClinVarStratifier,
    FDAPharmacovigilance,
    TMEMapper,
)


@pytest.mark.asyncio
async def test_fda_pharmacovigilance_query():
    fda = FDAPharmacovigilance()
    # Live openFDA FAERS API query
    report = await fda.fetch_safety_profile_async("tamoxifen", limit=3)
    assert report.drug_name == "Tamoxifen"
    assert report.total_patient_reports > 0
    assert len(report.top_adverse_reactions) > 0
    assert "SAFETY" in report.clinical_safety_verdict or "RISK" in report.clinical_safety_verdict


@pytest.mark.asyncio
async def test_clinvar_genomic_stratification():
    stratifier = ClinVarStratifier()
    report = await stratifier.stratify_target_async("EGFR", target_condition="Glioblastoma", max_variants=2)
    assert report.target_gene == "EGFR"
    assert report.total_pathogenic_variants_in_db > 0
    assert len(report.top_pathogenic_variants) > 0
    assert "CLINICAL NGS CRITERIA" in report.clinical_trial_genomic_criteria


def test_single_cell_tme_deconvolution():
    report = TMEMapper.analyze_microenvironment(["CD274", "STAT3", "TGFB1"], condition="Glioblastoma")
    assert report.target_disease == "Glioblastoma"
    assert len(report.target_expression_by_cell_type) >= 4
    assert len(report.active_ligand_receptor_axes) > 0
    assert report.stromal_immunosuppression_index > 50.0
    assert "IMMUNE_" in report.microenvironment_subtype
