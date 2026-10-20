"""Ollama API Client for HelixMind.

Provides asynchronous and synchronous methods to communicate with local Ollama
instances for chat completions, streaming, structured outputs, and embeddings.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Type, TypeVar, Union
import httpx
from pydantic import BaseModel

from helixmind.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class OllamaError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when connecting to Ollama server fails."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the specified model is not found on the local Ollama host."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an error response."""
    pass


class OllamaClient:
    """Client for Ollama REST API supporting chat, streaming, and embeddings."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        default_model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        cfg = settings or get_settings()
        self.base_url = (base_url or cfg.ollama_base_url).rstrip("/")
        self.timeout = timeout or cfg.ollama_timeout
        self.default_model = default_model or cfg.ollama_default_model
        self.reasoning_model = reasoning_model or cfg.ollama_reasoning_model
        self.embedding_model = embedding_model or cfg.ollama_embedding_model

    def _sync_client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    # -------------------------------------------------------------------------
    # Health and Model Management
    # -------------------------------------------------------------------------

    def check_health(self) -> bool:
        """Check if Ollama server is running and reachable."""
        try:
            with self._sync_client() as client:
                res = client.get("/api/version")
                return res.status_code == 200
        except Exception:
            return False

    async def check_health_async(self) -> bool:
        """Asynchronously check if Ollama server is reachable."""
        try:
            async with self._async_client() as client:
                res = await client.get("/api/version")
                return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models installed locally in Ollama."""
        try:
            with self._sync_client() as client:
                res = client.get("/api/tags")
                res.raise_for_status()
                data = res.json()
                return data.get("models", [])
        except httpx.RequestError as e:
            raise OllamaConnectionError(f"Could not connect to Ollama at {self.base_url}: {e}") from e
        except Exception as e:
            raise OllamaResponseError(f"Error listing Ollama models: {e}") from e

    async def list_models_async(self) -> List[Dict[str, Any]]:
        """Asynchronously list all local models."""
        try:
            async with self._async_client() as client:
                res = await client.get("/api/tags")
                res.raise_for_status()
                data = res.json()
                return data.get("models", [])
        except httpx.RequestError as e:
            raise OllamaConnectionError(f"Could not connect to Ollama at {self.base_url}: {e}") from e
        except Exception as e:
            raise OllamaResponseError(f"Error listing Ollama models: {e}") from e

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model or model tag is available locally."""
        models = self.list_models()
        model_names = [m.get("name", "") for m in models]
        # Match exact name or name without :latest if requested without tag
        for name in model_names:
            if name == model_name or name.split(":")[0] == model_name:
                return True
        return False

    # -------------------------------------------------------------------------
    # Chat Completions
    # -------------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send chat messages and return the full response."""
        chosen_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, **(options or {})},
        }
        if format:
            payload["format"] = format

        try:
            with self._sync_client() as client:
                res = client.post("/api/chat", json=payload)
                if res.status_code == 404:
                    raise OllamaModelNotFoundError(f"Model '{chosen_model}' not found in Ollama.")
                res.raise_for_status()
                return res.json()
        except httpx.RequestError as e:
            raise OllamaConnectionError(f"Connection failed to Ollama at {self.base_url}: {e}") from e

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronously send chat messages and return the full response."""
        chosen_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, **(options or {})},
        }
        if format:
            payload["format"] = format

        try:
            async with self._async_client() as client:
                res = await client.post("/api/chat", json=payload)
                if res.status_code == 404:
                    raise OllamaModelNotFoundError(f"Model '{chosen_model}' not found in Ollama.")
                res.raise_for_status()
                return res.json()
        except httpx.RequestError as e:
            raise OllamaConnectionError(f"Connection failed to Ollama at {self.base_url}: {e}") from e

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Stream response tokens synchronously."""
        chosen_model = model or self.default_model
        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, **(options or {})},
        }

        with self._sync_client() as client:
            with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content

    async def chat_stream_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """Stream response tokens asynchronously."""
        chosen_model = model or self.default_model
        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, **(options or {})},
        }

        async with self._async_client() as client:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content

    # -------------------------------------------------------------------------
    # Structured Output with Pydantic
    # -------------------------------------------------------------------------

    def chat_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Request structured JSON completion and parse into a Pydantic model."""
        # Append schema instruction
        system_instruction = (
            f"\nYou must return ONLY a valid JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"Do not include any conversational text, explanations, or codeblock fences."
        )
        augmented_messages = list(messages)
        if augmented_messages and augmented_messages[0].get("role") == "system":
            augmented_messages[0] = {
                "role": "system",
                "content": augmented_messages[0]["content"] + system_instruction,
            }
        else:
            augmented_messages.insert(0, {"role": "system", "content": system_instruction})

        response = self.chat(
            messages=augmented_messages,
            model=model,
            temperature=temperature,
            format="json",
        )
        raw_text = response.get("message", {}).get("content", "")
        clean_json = self._clean_json_text(raw_text)
        return schema.model_validate_json(clean_json)

    async def chat_structured_async(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Asynchronously request structured JSON completion and parse into a Pydantic model."""
        system_instruction = (
            f"\nYou must return ONLY a valid JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"Do not include any conversational text, explanations, or codeblock fences."
        )
        augmented_messages = list(messages)
        if augmented_messages and augmented_messages[0].get("role") == "system":
            augmented_messages[0] = {
                "role": "system",
                "content": augmented_messages[0]["content"] + system_instruction,
            }
        else:
            augmented_messages.insert(0, {"role": "system", "content": system_instruction})

        response = await self.chat_async(
            messages=augmented_messages,
            model=model,
            temperature=temperature,
            format="json",
        )
        raw_text = response.get("message", {}).get("content", "")
        clean_json = self._clean_json_text(raw_text)
        return schema.model_validate_json(clean_json)

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    def embed(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate vector embeddings for one or more text strings."""
        chosen_model = model or self.embedding_model
        input_texts = [texts] if isinstance(texts, str) else texts

        # Try /api/embed (standard in newer Ollama versions)
        payload = {"model": chosen_model, "input": input_texts}
        with self._sync_client() as client:
            res = client.post("/api/embed", json=payload)
            if res.status_code == 200:
                data = res.json()
                return data.get("embeddings", [])

            # Fallback to older /api/embeddings endpoint per text
            embeddings: List[List[float]] = []
            for text in input_texts:
                fb_res = client.post("/api/embeddings", json={"model": chosen_model, "prompt": text})
                fb_res.raise_for_status()
                embeddings.append(fb_res.json().get("embedding", []))
            return embeddings

    async def embed_async(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Asynchronously generate vector embeddings."""
        chosen_model = model or self.embedding_model
        input_texts = [texts] if isinstance(texts, str) else texts

        payload = {"model": chosen_model, "input": input_texts}
        async with self._async_client() as client:
            res = await client.post("/api/embed", json=payload)
            if res.status_code == 200:
                data = res.json()
                return data.get("embeddings", [])

            # Fallback
            embeddings: List[List[float]] = []
            for text in input_texts:
                fb_res = await client.post("/api/embeddings", json={"model": chosen_model, "prompt": text})
                fb_res.raise_for_status()
                embeddings.append(fb_res.json().get("embedding", []))
            return embeddings

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Strip markdown fences or extra reasoning text around JSON."""
        # Extract markdown json block if present
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()
        # Fallback to finding outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text.strip()
