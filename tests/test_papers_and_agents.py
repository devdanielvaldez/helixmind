"""Unit and integration tests for paper clients, vector RAG indexer, and dynamic agents."""

from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from helixmind.agents.factory import AgentFactory, SearchStrategy
from helixmind.papers.clinical_trials_client import ClinicalTrialsClient
from helixmind.papers.europe_pmc_client import EuropePMCClient
from helixmind.papers.indexer import SemanticIndex
from helixmind.papers.models import Paper
from helixmind.papers.ncbi_client import NCBIClient


@pytest.mark.asyncio
async def test_ncbi_search_and_fetch():
    client = NCBIClient()
    pmids = await client.search_pubmed_async("Huntington disease huntingtin", limit=2)
    assert isinstance(pmids, list)
    if pmids:
        papers = await client.fetch_papers_async(pmids)
        assert len(papers) > 0
        assert papers[0].title != ""
        assert papers[0].abstract != ""


def test_ncbi_parse_xml_structured_abstract():
    sample_xml = """<?xml version="1.0"?>
    <!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345678</PMID>
                <Article>
                    <ArticleTitle>Test Article</ArticleTitle>
                    <Abstract>
                        <AbstractText Label="BACKGROUND">Background on <i>in vivo</i> models.</AbstractText>
                        <AbstractText Label="RESULTS">Target inhibition achieved.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """
    abstracts = NCBIClient._parse_pubmed_xml(sample_xml)
    assert "12345678" in abstracts
    assert "BACKGROUND: Background on in vivo models." in abstracts["12345678"]
    assert "RESULTS: Target inhibition achieved." in abstracts["12345678"]


@pytest.mark.asyncio
async def test_ncbi_batch_chunking_avoids_414():
    """Verify that fetch_papers_async chunks PMIDs into batches of 200 and uses POST to prevent HTTP 414."""
    client = NCBIClient()
    # 250 dummy PMIDs to verify chunking and POST requests
    fake_pmids = [str(10000000 + i) for i in range(250)]

    mock_summary_res = {
        "result": {
            p: {
                "title": f"Paper Title {p}",
                "source": "Journal of Oncology",
                "pubdate": "2025 Jan",
                "authors": [{"name": "Smith J"}],
                "articleids": [{"idtype": "doi", "value": f"10.1000/{p}"}],
                "pmcrefcount": 5,
            }
            for p in fake_pmids
        }
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Mock responses for esummary and efetch
        mock_resp_summary = MagicMock(spec=httpx.Response)
        mock_resp_summary.status_code = 200
        mock_resp_summary.raise_for_status.return_value = None
        mock_resp_summary.json.return_value = mock_summary_res

        mock_resp_fetch = MagicMock(spec=httpx.Response)
        mock_resp_fetch.status_code = 200
        mock_resp_fetch.text = "<PubmedArticleSet></PubmedArticleSet>"

        mock_post.side_effect = [
            mock_resp_summary, mock_resp_fetch,  # batch 1 (0..200)
            mock_resp_summary, mock_resp_fetch,  # batch 2 (200..250)
        ]

        papers = await client.fetch_papers_async(fake_pmids)
        assert len(papers) == 250
        assert mock_post.call_count == 4  # 2 batches * 2 calls (esummary + efetch)
        # Verify POST was used with form data containing IDs, not URL parameters
        for call_args in mock_post.call_args_list:
            assert "data" in call_args.kwargs
            assert "id" in call_args.kwargs["data"]
            assert len(call_args.kwargs["data"]["id"].split(",")) <= 200


@pytest.mark.asyncio
async def test_europe_pmc_search():
    client = EuropePMCClient()
    papers = await client.search_async("CRISPR Cas9 oncology", limit=2)
    assert len(papers) > 0
    assert papers[0].source in ["europe_pmc", "preprint", "MED"]


@pytest.mark.asyncio
async def test_clinical_trials_search():
    client = ClinicalTrialsClient()
    studies = await client.search_studies_async(condition="Alzheimer", limit=2)
    assert isinstance(studies, list)
    if studies:
        assert studies[0].nct_id.startswith("NCT")


@pytest.mark.asyncio
async def test_semantic_index_rag():
    index = SemanticIndex()
    p1 = Paper(
        id="test_1",
        title="Microglial activation in Alzheimer's Disease",
        abstract="TREM2 signaling modulates microglial phagocytosis of amyloid beta plaques in brain tissue.",
    )
    p2 = Paper(
        id="test_2",
        title="KRAS G12D inhibition in Pancreatic Cancer",
        abstract="Small molecule MRTX1133 selectively binds and inhibits KRAS G12D in pancreatic ductal adenocarcinoma.",
    )

    indexed = await index.index_documents_async([p1, p2])
    assert indexed > 0

    # Test retrieval with semantic query
    results = await index.search_async("TREM2 microglial phagocytosis", top_k=1)
    assert len(results) > 0
    best_chunk, score = results[0]
    assert best_chunk.doc_id == "test_1"
    assert score > 0.4


@pytest.mark.asyncio
async def test_agent_spawner():
    factory = AgentFactory()
    strategist = factory.create_search_strategist()
    assert strategist.name == "Dr. Strategist"

    mock_strat = SearchStrategy(
        primary_condition="triple negative breast cancer",
        pubmed_query='"triple negative breast neoplasms"[Title/Abstract] AND immunotherapy',
        clinical_trials_condition="triple negative breast cancer",
        europe_pmc_query="triple negative breast cancer immunotherapy",
        key_biological_targets=["PD-L1", "TROP-2"],
    )

    with patch.object(strategist, "execute_structured_task_async", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_strat
        strategy = await strategist.execute_structured_task_async(
            task_instruction="Formulate search terms for: Triple negative breast cancer immunotherapy",
            schema=SearchStrategy,
        )
        assert strategy.primary_condition == "triple negative breast cancer"
        assert "PD-L1" in strategy.key_biological_targets
