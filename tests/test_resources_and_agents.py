"""Unit tests for SystemResourceGuard, thread allocation, limit sanitization, and specialized agent creation."""

import pytest
from unittest.mock import patch, MagicMock

from helixmind.agents.factory import AgentFactory
from helixmind.config import Settings
from helixmind.core.resources import SystemResourceGuard, get_resource_manager
from helixmind.llm.ollama_client import OllamaClient


def test_resource_guard_singleton():
    rm1 = get_resource_manager()
    rm2 = get_resource_manager()
    assert rm1 is rm2


def test_resource_guard_thread_capping():
    settings = Settings(enable_resource_guard=True)
    guard = SystemResourceGuard(settings=settings)

    threads = guard.get_safe_thread_count()
    assert threads >= 1
    assert threads <= guard.total_cpu_cores
    # Always reserves at least 1-2 cores if total cores >= 2
    if guard.total_cpu_cores >= 2:
        assert threads < guard.total_cpu_cores


def test_resource_guard_memory_metrics():
    guard = SystemResourceGuard()
    mem = guard.get_memory_info()

    assert "total_gb" in mem
    assert "available_gb" in mem
    assert "used_gb" in mem
    assert "percent_used" in mem
    assert mem["total_gb"] > 0
    assert 0.0 <= mem["percent_used"] <= 100.0


def test_resource_guard_sanitize_limits_clamping():
    settings = Settings(max_safe_papers_limit=100, max_safe_trials_limit=50)
    guard = SystemResourceGuard(settings=settings)

    # Within limits: no clamping, no warning
    p, t, w = guard.sanitize_request_limits(20, 10)
    assert p == 20
    assert t == 10
    assert w is None

    # Exceeding limits (e.g. 10,000 papers and 10,000 trials as requested by user)
    p_clamped, t_clamped, warning = guard.sanitize_request_limits(10000, 10000)
    assert p_clamped == 100
    assert t_clamped == 50
    assert warning is not None
    assert "Resource Guard" in warning
    assert "10,000" in warning
    assert "freeze" in warning.lower() or "ram" in warning.lower()


def test_resource_guard_safe_ollama_options():
    settings = Settings(enable_resource_guard=True, max_context_tokens=4096)
    guard = SystemResourceGuard(settings=settings)

    opts = guard.get_safe_ollama_options({"temperature": 0.3})
    assert "num_thread" in opts
    assert opts["num_ctx"] == 4096
    assert opts["temperature"] == 0.3


@pytest.mark.asyncio
async def test_resource_guard_throttle_and_gc():
    guard = SystemResourceGuard()
    collected = guard.optimize_memory()
    assert isinstance(collected, int)

    # Throttling yields without error
    await guard.throttle_if_needed(extra_pause=0.01)


def test_ollama_client_automatic_safe_options():
    client = OllamaClient()
    built = client._build_options(temperature=0.2)
    assert "num_thread" in built
    assert "num_ctx" in built
    assert built["temperature"] == 0.2


def test_specialized_agents_creation():
    client = OllamaClient()
    factory = AgentFactory(ollama_client=client)

    # 1. Dr. Librarian (Evidence Curator)
    librarian = factory.create_evidence_curator()
    assert librarian.name == "Dr. Librarian"
    assert "Curator" in librarian.role or "Biomedical" in librarian.role
    assert librarian.temperature == 0.2

    # 2. Dr. Graph (Knowledge Graph Specialist)
    graph_agent = factory.create_knowledge_graph_specialist()
    assert graph_agent.name == "Dr. Graph"
    assert "Graph" in graph_agent.role or "Systems Biology" in graph_agent.role

    # 3. Dr. DepMap (Cancer Dependency)
    depmap_agent = factory.create_depmap_specialist()
    assert depmap_agent.name == "Dr. DepMap"
    assert "DepMap" in depmap_agent.name

    # 4. Dr. Pharmacovigilance (openFDA FAERS)
    pv_agent = factory.create_pharmacovigilance_specialist()
    assert pv_agent.name == "Dr. Pharmacovigilance"
    assert "Safety" in pv_agent.role or "Surveillance" in pv_agent.role

    # 5. Dr. ClinVar (Genomic Pathologist)
    clinvar_agent = factory.create_clinvar_specialist()
    assert clinvar_agent.name == "Dr. ClinVar"
    assert "ClinVar" in clinvar_agent.name

    # 6. Dr. TME (Single-Cell Microenvironment)
    tme_agent = factory.create_tme_specialist()
    assert tme_agent.name == "Dr. TME"
    assert "Microenvironment" in tme_agent.role

    # 7. Dr. CRISPR (Genome Editing)
    crispr_agent = factory.create_crispr_specialist()
    assert crispr_agent.name == "Dr. CRISPR"
    assert "CRISPR" in crispr_agent.name or "Genome" in crispr_agent.role

    # 8. Dr. Editor (Lead Scientific Author)
    editor_agent = factory.create_paper_editor()
    assert editor_agent.name == "Dr. Editor"
    assert "Editor" in editor_agent.role or "Author" in editor_agent.role
