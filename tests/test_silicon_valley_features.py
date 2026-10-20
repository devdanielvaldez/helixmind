"""Unit tests for DepMap verification, De Novo chemistry, and Virtual Patient Trial Simulation."""

from unittest.mock import AsyncMock, patch
import pytest
from helixmind.insilico import (
    ClinicalSimulator,
    DeNovoGenerator,
    DepMapVerifier,
)
from rdkit import Chem


@pytest.mark.asyncio
async def test_depmap_essentiality_verifier():
    verifier = DepMapVerifier()
    res = await verifier.verify_gene_essentiality_async("STAT3", target_condition="Glioblastoma")
    assert res.gene_symbol == "STAT3"
    assert res.chronos_score <= -0.5
    assert res.essentiality_tier in ["STRONGLY_SELECTIVE", "PAN_CANCER_ESSENTIAL", "MODERATE_DEPENDENCY"]
    assert len(res.top_dependent_cell_lines) > 0


def test_denovo_chemical_generation_and_sascore():
    gen = DeNovoGenerator()
    analogs = gen.generate_optimized_analogs(
        parent_drug_name="tamoxifen",
        max_analogs=2,
    )
    assert len(analogs) > 0
    a1 = analogs[0]
    assert a1.compound_code.startswith("HM-NCE-")
    assert 1.0 <= a1.sa_score <= 10.0
    assert "SYNTHESIZABLE" in a1.sa_verdict or "SYNTHESIS" in a1.sa_verdict
    assert isinstance(a1.bbb_penetrant, bool)
    assert a1.herg_cardiotox_risk in ["LOW", "MEDIUM", "HIGH"]

    # Verify RDKit validity of generated analog
    mol = Chem.MolFromSmiles(a1.smiles)
    assert mol is not None


def test_clinical_trial_simulator_kaplan_meier():
    sim_report = ClinicalSimulator.simulate_trial(
        drug_name="HM-NCE-01",
        target_condition="Glioblastoma",
        predicted_delta_g=-8.8,
        cohort_size=150,
        random_seed=42,
    )

    assert sim_report.total_patients_simulated == 300
    assert 0.2 <= sim_report.hazard_ratio <= 1.0
    assert sim_report.median_pfs_experimental_months > 0.0
    assert sim_report.probability_of_technical_success > 50.0
    assert len(sim_report.kaplan_meier_curve) >= 6
    # Survival should decline monotonically
    surv_rates = [p.experimental_survival_rate for p in sim_report.kaplan_meier_curve]
    assert surv_rates[0] == 1.0
    assert surv_rates[-1] < surv_rates[0]
