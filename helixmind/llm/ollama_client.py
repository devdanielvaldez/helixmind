"""Ollama API Client for HelixMind.

Provides asynchronous and synchronous methods to communicate with local Ollama
instances for chat completions, streaming, structured outputs, and embeddings.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
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
        return httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(self.timeout, connect=30.0))

    def _async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(self.timeout, connect=30.0))

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

        for attempt in range(3):
            try:
                with self._sync_client() as client:
                    res = client.post("/api/chat", json=payload)
                    if res.status_code == 404:
                        raise OllamaModelNotFoundError(f"Model '{chosen_model}' not found in Ollama.")
                    res.raise_for_status()
                    return res.json()
            except httpx.RequestError as e:
                if attempt < 2:
                    time.sleep(2.0)
                    continue
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

        for attempt in range(3):
            try:
                async with self._async_client() as client:
                    res = await client.post("/api/chat", json=payload)
                    if res.status_code == 404:
                        raise OllamaModelNotFoundError(f"Model '{chosen_model}' not found in Ollama.")
                    res.raise_for_status()
                    return res.json()
            except httpx.RequestError as e:
                if attempt < 2:
                    await asyncio.sleep(2.0)
                    continue
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
    # Multi-Part and Chunked Large Output Generation
    # -------------------------------------------------------------------------

    async def chat_with_continuation_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
        max_continuations: int = 3,
    ) -> str:
        """Send chat request and automatically continue generation if truncated by max token length."""
        accumulated_text = ""
        current_messages = list(messages)

        for _ in range(max_continuations + 1):
            res = await self.chat_async(
                messages=current_messages,
                model=model,
                temperature=temperature,
                options=options,
            )
            content = res.get("message", {}).get("content", "")
            accumulated_text += content

            # Check if terminated by length (Ollama output limit reached)
            if res.get("done_reason") == "length":
                current_messages.append({"role": "assistant", "content": content})
                current_messages.append({
                    "role": "user",
                    "content": "Continue immediately from the exact point you stopped. Do not repeat previous text.",
                })
            else:
                break

        return accumulated_text

    async def chat_multipart_async(
        self,
        goal: str,
        sections: List[Dict[str, str]],
        context: str = "",
        model: Optional[str] = None,
        temperature: float = 0.5,
    ) -> Dict[str, str]:
        """Generate a comprehensive multi-part response section by section to avoid token truncation."""
        assembled_sections: Dict[str, str] = {}
        rolling_summary = ""

        for i, sec in enumerate(sections, 1):
            title = sec.get("title", f"Part {i}")
            instruction = sec.get("prompt", "")

            section_prompt = (
                f"### Overarching Scientific Goal:\n{goal}\n\n"
                f"### Empirical Context:\n{context[:1000]}\n\n"
            )
            if rolling_summary:
                section_prompt += f"### Previous Sections Completed:\n{rolling_summary}\n\n"

            section_prompt += (
                f"### Current Task: Write Section {i} - '{title}'\n"
                f"Instruction: {instruction}\n"
                f"Write an exhaustive, thorough explanation for this section only."
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are HelixMind, an exhaustive scientific intelligence. Provide deep, rigorous content for the requested section.",
                },
                {"role": "user", "content": section_prompt},
            ]

            content = await self.chat_with_continuation_async(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            assembled_sections[title] = content.strip()
            rolling_summary += f"- {title}: {content[:150].strip()}...\n"

        return assembled_sections

    # -------------------------------------------------------------------------
    # Structured Output with Pydantic
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_schema_instruction(schema: Type[T]) -> str:
        properties = schema.model_json_schema().get("properties", {})
        example = {}
        for key, prop in properties.items():
            desc = prop.get("description", prop.get("type", "value"))
            example[key] = f"<{desc}>"
        return (
            f"\nYou must return ONLY a valid JSON object populated with actual values according to these keys:\n"
            f"{json.dumps(example, indent=2)}\n"
            f"Do NOT return the schema or type definitions. Return concrete filled values."
        )

    def chat_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Request structured JSON completion and parse into a Pydantic model."""
        system_instruction = self._build_schema_instruction(schema)
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
        try:
            return schema.model_validate_json(clean_json)
        except Exception:
            parsed = json.loads(clean_json)
            if isinstance(parsed, list):
                # If wrapped schema has a list field, wrap it
                fields = schema.model_fields
                list_keys = [k for k, f in fields.items() if getattr(f.annotation, "__origin__", None) is list or "List" in str(f.annotation)]
                if list_keys:
                    return schema.model_validate({list_keys[0]: parsed})
            elif isinstance(parsed, dict):
                filtered = {
                    k: v for k, v in parsed.items()
                    if not (isinstance(v, str) and (v.startswith("<") or v in ["string", "object"]))
                }
                return schema.model_validate(filtered)
            raise

    async def chat_structured_async(
        self,
        messages: List[Dict[str, str]],
        schema: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        """Asynchronously request structured JSON completion and parse into a Pydantic model."""
        system_instruction = self._build_schema_instruction(schema)
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
        try:
            return schema.model_validate_json(clean_json)
        except Exception:
            parsed = json.loads(clean_json)
            if isinstance(parsed, list):
                fields = schema.model_fields
                list_keys = [k for k, f in fields.items() if getattr(f.annotation, "__origin__", None) is list or "List" in str(f.annotation)]
                if list_keys:
                    return schema.model_validate({list_keys[0]: parsed})
            elif isinstance(parsed, dict):
                filtered = {
                    k: v for k, v in parsed.items()
                    if not (isinstance(v, str) and (v.startswith("<") or v in ["string", "object"]))
                }
                return schema.model_validate(filtered)
            raise

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
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()

        first_curly = text.find("{")
        last_curly = text.rfind("}")
        first_square = text.find("[")
        last_square = text.rfind("]")

        if first_square != -1 and (first_curly == -1 or first_square < first_curly) and last_square > first_square:
            return text[first_square : last_square + 1]
        elif first_curly != -1 and last_curly > first_curly:
            return text[first_curly : last_curly + 1]
        return text.strip()
