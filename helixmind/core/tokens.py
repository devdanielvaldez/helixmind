"""Real-time LLM Token Tracker and Inference Speed Analytics for HelixMind.

Tracks prompt tokens, completion tokens, inference duration, tokens per second,
and equivalent cost savings across all multi-agent cycles and investigations.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class TokenRecord(BaseModel):
    """Aggregate token and throughput statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_duration_sec: float = 0.0
    eval_duration_sec: float = 0.0
    calls_count: int = 0
    last_speed_tps: float = 0.0
    average_speed_tps: float = 0.0


class TokenTracker:
    """Thread-safe centralized token and inference performance tracker."""

    _instance: Optional[TokenTracker] = None
    _lock = threading.Lock()

    def __new__(cls) -> TokenTracker:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_tracker()
            return cls._instance

    def _init_tracker(self) -> None:
        self.lock = threading.Lock()
        self.total = TokenRecord()
        self.by_investigation: Dict[str, TokenRecord] = {}
        self.by_model: Dict[str, TokenRecord] = {}
        self.by_agent: Dict[str, TokenRecord] = {}
        self.active_investigation_id: Optional[str] = None
        self._load_persisted_state()

    def set_active_investigation(self, investigation_id: Optional[str]) -> None:
        """Bind the tracker to a current investigation ID."""
        with self.lock:
            self.active_investigation_id = investigation_id

    def record(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "default",
        eval_duration_ns: int = 0,
        total_duration_ns: int = 0,
        investigation_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Record token usage and timing from an Ollama response."""
        total_toks = prompt_tokens + completion_tokens
        eval_sec = eval_duration_ns / 1e9 if eval_duration_ns > 0 else 0.0
        total_sec = total_duration_ns / 1e9 if total_duration_ns > 0 else 0.0

        current_tps = (completion_tokens / eval_sec) if eval_sec > 0 and completion_tokens > 0 else 0.0
        inv_id = investigation_id or self.active_investigation_id or "global"

        with self.lock:
            # 1. Update Global Aggregate
            self._update_record(
                self.total,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                eval_sec=eval_sec,
                total_sec=total_sec,
                current_tps=current_tps,
            )

            # 2. Update Per-Investigation
            if inv_id not in self.by_investigation:
                self.by_investigation[inv_id] = TokenRecord()
            self._update_record(
                self.by_investigation[inv_id],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                eval_sec=eval_sec,
                total_sec=total_sec,
                current_tps=current_tps,
            )

            # 3. Update Per-Model
            clean_model = model or "default"
            if clean_model not in self.by_model:
                self.by_model[clean_model] = TokenRecord()
            self._update_record(
                self.by_model[clean_model],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                eval_sec=eval_sec,
                total_sec=total_sec,
                current_tps=current_tps,
            )

            # 4. Update Per-Agent
            if agent:
                if agent not in self.by_agent:
                    self.by_agent[agent] = TokenRecord()
                self._update_record(
                    self.by_agent[agent],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    eval_sec=eval_sec,
                    total_sec=total_sec,
                    current_tps=current_tps,
                )

        self._save_persisted_state()

    def _update_record(
        self,
        record: TokenRecord,
        prompt_tokens: int,
        completion_tokens: int,
        eval_sec: float,
        total_sec: float,
        current_tps: float,
    ) -> None:
        record.prompt_tokens += prompt_tokens
        record.completion_tokens += completion_tokens
        record.total_tokens += (prompt_tokens + completion_tokens)
        record.eval_duration_sec += eval_sec
        record.total_duration_sec += total_sec
        record.calls_count += 1
        if current_tps > 0:
            record.last_speed_tps = round(current_tps, 1)
        if record.eval_duration_sec > 0 and record.completion_tokens > 0:
            record.average_speed_tps = round(record.completion_tokens / record.eval_duration_sec, 1)

    def get_metrics(self, investigation_id: Optional[str] = None) -> Dict[str, Any]:
        """Return a structured summary of token consumption and speed metrics."""
        with self.lock:
            target_record = self.total
            target_id = investigation_id or self.active_investigation_id
            if target_id and target_id in self.by_investigation:
                target_record = self.by_investigation[target_id]

            # Commercial API cost equivalent estimation (e.g., $3/M prompt, $15/M completion)
            est_cost_usd = (
                (target_record.prompt_tokens / 1_000_000.0) * 3.0
                + (target_record.completion_tokens / 1_000_000.0) * 15.0
            )

            return {
                "investigation_id": target_id or "global",
                "prompt_tokens": target_record.prompt_tokens,
                "completion_tokens": target_record.completion_tokens,
                "total_tokens": target_record.total_tokens,
                "calls_count": target_record.calls_count,
                "last_speed_tps": target_record.last_speed_tps,
                "average_speed_tps": target_record.average_speed_tps,
                "total_duration_sec": round(target_record.total_duration_sec, 2),
                "eval_duration_sec": round(target_record.eval_duration_sec, 2),
                "estimated_commercial_value_usd": round(est_cost_usd, 4),
                "global_total_tokens": self.total.total_tokens,
                "models_breakdown": {
                    m: rec.model_dump() for m, rec in self.by_model.items()
                },
            }

    def _state_file_path(self) -> Path:
        return Path("data") / "tokens_state.json"

    def _save_persisted_state(self) -> None:
        try:
            path = self._state_file_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total": self.total.model_dump(),
                "by_investigation": {k: v.model_dump() for k, v in self.by_investigation.items()},
                "by_model": {k: v.model_dump() for k, v in self.by_model.items()},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_persisted_state(self) -> None:
        try:
            path = self._state_file_path()
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "total" in data:
                        self.total = TokenRecord(**data["total"])
                    if "by_investigation" in data:
                        for k, v in data["by_investigation"].items():
                            self.by_investigation[k] = TokenRecord(**v)
                    if "by_model" in data:
                        for k, v in data["by_model"].items():
                            self.by_model[k] = TokenRecord(**v)
        except Exception:
            pass

    def reset(self) -> None:
        """Reset all tracked token counts."""
        with self.lock:
            self.total = TokenRecord()
            self.by_investigation.clear()
            self.by_model.clear()
            self.by_agent.clear()
        self._save_persisted_state()


def get_token_tracker() -> TokenTracker:
    """Return the global TokenTracker singleton."""
    return TokenTracker()
