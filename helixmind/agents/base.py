"""Base Agent definition for HelixMind multi-agent system."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel
from rich.console import Console

from helixmind.config import get_settings
from helixmind.llm.ollama_client import OllamaClient

T = TypeVar("T", bound=BaseModel)
console = Console()


class BaseAgent:
    """Autonomous agent equipped with role persona, memory, and LLM inference."""

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.5,
        ollama_client: Optional[OllamaClient] = None,
    ):
        settings = get_settings()
        self.name = name
        self.role = role
        self.goal = goal
        self.system_prompt = system_prompt
        self.model = model or settings.ollama_default_model
        self.temperature = temperature
        self.client = ollama_client or OllamaClient()
        self.memory: List[Dict[str, str]] = []

    def log(self, message: str, style: str = "cyan") -> None:
        """Log agent activity to console."""
        console.print(f"[{style}][{self.name} - {self.role}]:[/] {message}")

    async def execute_task_async(
        self,
        task_instruction: str,
        context: str = "",
        temperature: Optional[float] = None,
    ) -> str:
        """Execute a cognitive task and return the generated response."""
        full_user_prompt = task_instruction
        if context:
            full_user_prompt += f"\n\n### Scientific Context & Evidence:\n{context}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory,
            {"role": "user", "content": full_user_prompt},
        ]

        temp = temperature if temperature is not None else self.temperature
        reply = await self.client.chat_with_continuation_async(
            messages=messages,
            model=self.model,
            temperature=temp,
        )

        # Record conversation in short-term memory
        self.memory.append({"role": "user", "content": task_instruction})
        self.memory.append({"role": "assistant", "content": reply})
        return reply

    async def execute_multipart_task_async(
        self,
        goal: str,
        sections: List[Dict[str, str]],
        context: str = "",
        temperature: Optional[float] = None,
    ) -> Dict[str, str]:
        """Execute a cognitive task in progressive parts/sections to generate exhaustive content without truncation."""
        temp = temperature if temperature is not None else self.temperature
        self.log(f"Beginning multi-part synthesis across {len(sections)} distinct sections...", style="bold cyan")

        results: Dict[str, str] = {}
        rolling_context = ""

        for i, sec in enumerate(sections, 1):
            title = sec.get("title", f"Section {i}")
            instruction = sec.get("prompt", "")
            self.log(f"Generating Part {i}/{len(sections)}: '{title}'...", style="dim cyan")

            section_prompt = (
                f"### Overarching Investigation Goal: {goal}\n\n"
                f"### Empirical Evidence:\n{context[:1000]}\n\n"
            )
            if rolling_context:
                section_prompt += f"### Previous Completed Sections:\n{rolling_context}\n\n"

            section_prompt += (
                f"### Section {i} Focus: '{title}'\n"
                f"Instructions: {instruction}\n"
                f"Write deeply and thoroughly for this specific section only."
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": section_prompt},
            ]

            content = await self.client.chat_with_continuation_async(
                messages=messages,
                model=self.model,
                temperature=temp,
            )
            results[title] = content.strip()
            rolling_context += f"- **{title}**: {content[:150].strip()}...\n"

        self.log(f"Successfully completed all {len(sections)} sections.", style="bold green")
        return results

    async def execute_structured_task_async(
        self,
        task_instruction: str,
        schema: Type[T],
        context: str = "",
        temperature: float = 0.2,
    ) -> T:
        """Execute a cognitive task returning a validated Pydantic object."""
        full_user_prompt = task_instruction
        if context:
            full_user_prompt += f"\n\n### Scientific Context & Evidence:\n{context}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory,
            {"role": "user", "content": full_user_prompt},
        ]

        return await self.client.chat_structured_async(
            messages=messages,
            schema=schema,
            model=self.model,
            temperature=temperature,
        )
