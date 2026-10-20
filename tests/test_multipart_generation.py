"""Unit tests for multi-part chunked generation and auto-continuation."""

from unittest.mock import AsyncMock, patch
import pytest
from helixmind.agents.base import BaseAgent
from helixmind.llm.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_chat_continuation_logic():
    client = OllamaClient()

    # Mock Ollama returning first chunk with done_reason='length', second chunk with done_reason='stop'
    first_response = {"message": {"content": "First paragraph discussing cellular senescence. "}, "done_reason": "length"}
    second_response = {"message": {"content": "Second paragraph detailing SASP secretion."}, "done_reason": "stop"}

    with patch.object(client, "chat_async", side_effect=[first_response, second_response]):
        res = await client.chat_with_continuation_async(
            messages=[{"role": "user", "content": "Explain senescence"}],
            model="llama3.2:1b",
        )
        assert "First paragraph discussing cellular senescence." in res
        assert "Second paragraph detailing SASP secretion." in res


@pytest.mark.asyncio
async def test_agent_multipart_generation():
    client = OllamaClient()
    agent = BaseAgent(
        name="TestDr",
        role="Tester",
        goal="Test multipart",
        system_prompt="You are a scientific test assistant.",
        model="llama3.2:1b",
        ollama_client=client,
    )

    sections = [
        {"title": "Section A", "prompt": "State one sentence about DNA repair."},
        {"title": "Section B", "prompt": "State one sentence about apoptosis."},
    ]

    with patch.object(client, "chat_with_continuation_async", side_effect=["DNA repair pathway.", "Programmed cell death."]):
        parts = await agent.execute_multipart_task_async(
            goal="Cellular damage response",
            sections=sections,
        )

        assert "Section A" in parts
        assert "Section B" in parts
        assert parts["Section A"] == "DNA repair pathway."
        assert parts["Section B"] == "Programmed cell death."
