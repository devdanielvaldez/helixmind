"""Unit tests for multi-part chunked generation and auto-continuation."""

import pytest
from helixmind.agents.base import BaseAgent
from helixmind.llm.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_chat_continuation_logic():
    client = OllamaClient()
    # Test chat_with_continuation_async returns complete text
    res = await client.chat_with_continuation_async(
        messages=[{"role": "user", "content": "Explain cellular senescence in 2 sentences."}],
        model="llama3.2:1b",
    )
    assert isinstance(res, str)
    assert len(res) > 20


@pytest.mark.asyncio
async def test_agent_multipart_generation():
    client = OllamaClient()
    agent = BaseAgent(
        name="TestDr",
        role="Tester",
        goal="Test multipart",
        system_prompt="You are a scientific test assistant. Keep answers concise.",
        model="llama3.2:1b",
        ollama_client=client,
    )

    sections = [
        {"title": "Section A", "prompt": "State one sentence about DNA repair."},
        {"title": "Section B", "prompt": "State one sentence about apoptosis."},
    ]

    parts = await agent.execute_multipart_task_async(
        goal="Cellular damage response",
        sections=sections,
    )

    assert "Section A" in parts
    assert "Section B" in parts
    assert len(parts["Section A"]) > 10
    assert len(parts["Section B"]) > 10
