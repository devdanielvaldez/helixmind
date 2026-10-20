"""LLM interface and Ollama client integration."""

from helixmind.llm.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
)
from helixmind.llm.prompts import (
    SCIENTIFIC_SYSTEM_PROMPT,
    HYPOTHESIS_GENERATOR_PROMPT,
    CRITICAL_REVIEWER_PROMPT,
)

__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaResponseError",
    "SCIENTIFIC_SYSTEM_PROMPT",
    "HYPOTHESIS_GENERATOR_PROMPT",
    "CRITICAL_REVIEWER_PROMPT",
]

