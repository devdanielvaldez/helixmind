"""Unit tests for Research Director and dynamic agent mission execution."""

from unittest.mock import AsyncMock, patch
import pytest
from helixmind.agents.director import AgentMission, DirectorEvaluation, ResearchDirector
from helixmind.graph.models import KnowledgeGraph
from helixmind.journal.journal import ResearchJournal
from helixmind.papers.indexer import SemanticIndex
from helixmind.papers.models import Paper


def test_agent_mission_model():
    mission = AgentMission(
        agent_name="Dr. Immuno",
        agent_role="Immunogenomics Specialist",
        agent_goal="Analyze neoantigen signatures",
        system_prompt="You are Dr. Immuno",
        task_instruction="Examine MHC-I binding",
        rationale="Identify immunogenic candidates",
    )
    assert mission.agent_name == "Dr. Immuno"
    assert mission.task_type == "literature_deep_dive"


@pytest.mark.asyncio
async def test_director_dispatch_fallback():
    director = ResearchDirector(model="llama3.2:1b")
    journal = ResearchJournal(investigation_id="TEST-01", topic="Glioblastoma")
    graph = KnowledgeGraph()

    with patch("helixmind.agents.base.BaseAgent.execute_structured_task_async", new_callable=AsyncMock) as mock_exec:
        # Simulate structured response from Director
        mock_mission = AgentMission(
            agent_name="Dr. Epigenome",
            agent_role="Epigeneticist",
            agent_goal="Analyze DNA methylation",
            system_prompt="You are an epigeneticist.",
            task_instruction="Inspect MGMT promoter methylation",
            rationale="Key predictor of temozolomide response",
        )
        mock_exec.return_value = DirectorEvaluation(
            current_knowledge_assessment="MGMT status is critical.",
            unresolved_questions=["MGMT promoter methylation percentage"],
            agent_name="Dr. Epigenome",
            agent_role="Epigeneticist",
            agent_goal="Analyze DNA methylation",
            rationale="Key predictor of temozolomide response",
            investigation_ready_for_synthesis=False,
        )

        mission = await director.evaluate_and_dispatch_async(
            topic="Glioblastoma",
            cycle_number=1,
            max_cycles=2,
            journal=journal,
            knowledge_graph=graph,
            recent_findings_summary="MGMT is involved.",
        )

        assert mission.agent_name == "Dr. Epigenome"
        assert mission.agent_role == "Epigeneticist"


@pytest.mark.asyncio
async def test_execute_agent_mission_terminology():
    director = ResearchDirector(model="llama3.2:1b")
    journal = ResearchJournal(investigation_id="TEST-02", topic="Lung Cancer")
    graph = KnowledgeGraph()
    index = SemanticIndex()

    mission = AgentMission(
        agent_name="Dr. Lexicon",
        agent_role="Biomedical Ontologist",
        agent_goal="Define EGFR and KRAS",
        task_type="terminology_definition",
        system_prompt="You are Dr. Lexicon.",
        task_instruction="Define the terms.",
        rationale="Lexical clarity",
    )

    with patch("helixmind.agents.base.BaseAgent.execute_task_async", new_callable=AsyncMock) as mock_task:
        mock_task.return_value = "EGFR: Epidermal Growth Factor Receptor kinase\nKRAS: Kirsten rat sarcoma viral oncogene homolog"

        result = await director.execute_agent_mission_async(
            mission=mission,
            index=index,
            journal=journal,
            knowledge_graph=graph,
            ncbi_client=None,
            epmc_client=None,
            active_papers=[],
        )

        assert "EGFR" in result
        assert len(journal.agents_roster) == 1
        assert "egfr" in journal.glossary
        assert "kras" in journal.glossary
        assert "biomarker:egfr" in graph.nodes
