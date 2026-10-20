"""Unit and integration tests for Ollama client."""

import pytest
from pydantic import BaseModel
from helixmind.llm.ollama_client import OllamaClient


class MockScientificSummary(BaseModel):
    title: str
    key_finding: str


def test_clean_json_text_pure_json():
    raw = '{"title": "Test", "key_finding": "Result"}'
    cleaned = OllamaClient._clean_json_text(raw)
    assert cleaned == raw


def test_clean_json_text_with_markdown_fences():
    raw = '```json\n{"title": "Test", "key_finding": "Result"}\n```'
    cleaned = OllamaClient._clean_json_text(raw)
    assert cleaned == '{"title": "Test", "key_finding": "Result"}'


def test_clean_json_text_with_surrounding_commentary():
    raw = 'Here is the requested output:\n{"title": "Test", "key_finding": "Result"}\nHope this helps!'
    cleaned = OllamaClient._clean_json_text(raw)
    assert cleaned == '{"title": "Test", "key_finding": "Result"}'


def test_ollama_health_live():
    client = OllamaClient()
    is_alive = client.check_health()
    assert is_alive is True, "Ollama should be reachable on localhost:11434"


def test_ollama_list_models_live():
    client = OllamaClient()
    models = client.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    names = [m["name"] for m in models]
    # Check that at least one of our installed models is present
    assert any("llama" in n or "mistral" in n or "qwen" in n for n in names)


def test_ollama_embed_live():
    client = OllamaClient()
    if client.is_model_available("nomic-embed-text"):
        embeddings = client.embed("Target validation in oncology", model="nomic-embed-text:latest")
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
        assert isinstance(embeddings[0][0], float)


@pytest.mark.asyncio
async def test_ollama_async_health_live():
    client = OllamaClient()
    is_alive = await client.check_health_async()
    assert is_alive is True
