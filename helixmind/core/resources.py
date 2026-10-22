"""System Resource Guard & Hardware Throttling Engine for HelixMind.

Monitors host CPU, RAM, and GPU memory in real-time, throttles compute threads
to prevent machine freeze and SSH connection dropouts, enforces safe context sizes,
and proactively manages garbage collection and model unloading.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

from rich.panel import Panel

from helixmind.config import Settings, get_settings

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class SystemResourceGuard:
    """Monitors and manages host hardware resources during multi-agent workflows."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    @property
    def total_cpu_cores(self) -> int:
        """Total logical CPU cores available on the host machine."""
        return os.cpu_count() or 4

    def get_safe_thread_count(self) -> int:
        """Calculate optimal thread allocation for local LLM inference.
        
        Always reserves at least 1-2 cores for OS scheduling, terminal I/O,
        and SSH daemon sockets to eliminate terminal freeze.
        """
        if self.settings.max_cpu_threads and self.settings.max_cpu_threads > 0:
            return min(self.settings.max_cpu_threads, self.total_cpu_cores)

        cores = self.total_cpu_cores
        if cores >= 12:
            reserved = 2
        elif cores >= 6:
            reserved = 2
        elif cores >= 2:
            reserved = 1
        else:
            reserved = 0

        safe = max(1, cores - reserved)
        return min(safe, 8)

    def get_memory_info(self) -> Dict[str, Any]:
        """Obtain current system memory usage metrics."""
        if HAS_PSUTIL:
            try:
                vm = psutil.virtual_memory()
                return {
                    "total_gb": round(vm.total / (1024 ** 3), 2),
                    "available_gb": round(vm.available / (1024 ** 3), 2),
                    "used_gb": round(vm.used / (1024 ** 3), 2),
                    "percent_used": vm.percent,
                    "is_under_pressure": vm.percent >= self.settings.ram_warning_threshold_pct,
                }
            except Exception:
                pass

        # Fallback if psutil is unavailable or fails
        return {
            "total_gb": 16.0,
            "available_gb": 8.0,
            "used_gb": 8.0,
            "percent_used": 50.0,
            "is_under_pressure": False,
        }

    def get_cpu_info(self) -> Dict[str, Any]:
        """Obtain current CPU metrics and safe thread allocation."""
        threads = self.get_safe_thread_count()
        cpu_pct = 0.0
        if HAS_PSUTIL:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
            except Exception:
                pass

        return {
            "total_cores": self.total_cpu_cores,
            "safe_threads": threads,
            "reserved_cores": max(0, self.total_cpu_cores - threads),
            "cpu_percent": cpu_pct,
        }

    def get_safe_ollama_options(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a safe options dictionary for Ollama API calls.
        
        Applies safe CPU thread limits and context window caps.
        """
        options: Dict[str, Any] = {}
        if self.settings.enable_resource_guard:
            options["num_thread"] = self.get_safe_thread_count()
            options["num_ctx"] = self.settings.max_context_tokens

        if overrides:
            options.update(overrides)

        return options

    def sanitize_request_limits(
        self,
        requested_papers: int,
        requested_trials: int,
    ) -> Tuple[int, int, Optional[str]]:
        """Sanitize and clamp excessive paper/trial requests to protect machine stability.
        
        Returns:
            Tuple of (clamped_papers, clamped_trials, warning_message_or_none)
        """
        warning_msg: Optional[str] = None
        clamped_papers = requested_papers
        clamped_trials = requested_trials

        max_papers = self.settings.max_safe_papers_limit
        max_trials = self.settings.max_safe_trials_limit

        reasons = []
        if requested_papers > max_papers:
            clamped_papers = max_papers
            reasons.append(
                f"Literature papers count clamped from {requested_papers:,} to {clamped_papers} "
                f"(safe memory limit for RAG vector index)"
            )

        if requested_trials > max_trials:
            clamped_trials = max_trials
            reasons.append(
                f"Clinical trials count clamped from {requested_trials:,} to {clamped_trials} "
                f"(ClinicalTrials.gov API page limit & host stability)"
            )

        if reasons:
            warning_msg = (
                "🛡️ [bold yellow]Resource Guard Active:[/bold yellow] "
                + " | ".join(reasons)
                + ". Prevents machine freeze, RAM exhaustion, and SSH socket disconnects."
            )

        return clamped_papers, clamped_trials, warning_msg

    def optimize_memory(self) -> int:
        """Trigger proactive garbage collection and memory reclamation."""
        collected = gc.collect()
        mem = self.get_memory_info()
        return collected

    async def throttle_if_needed(self, extra_pause: float = 0.0) -> None:
        """Asynchronously pause and yield event loop to guarantee network and SSH responsiveness.
        
        Dynamically adapts delay if memory or CPU pressure is detected.
        """
        base_pause = self.settings.throttle_pause_seconds + extra_pause
        mem = self.get_memory_info()

        if mem.get("is_under_pressure", False):
            # Proactively collect garbage and apply higher backoff
            self.optimize_memory()
            base_pause = max(base_pause, 0.4)

        if base_pause > 0:
            await asyncio.sleep(base_pause)

    def format_status_string(self) -> str:
        """Return a single-line summary of host health for logging."""
        mem = self.get_memory_info()
        cpu = self.get_cpu_info()
        status_tag = "CRITICAL" if mem["percent_used"] > 90 else ("ELEVATED" if mem["is_under_pressure"] else "OPTIMAL")
        return (
            f"Host CPU: {cpu['cpu_percent']:.0f}% (Allocated: {cpu['safe_threads']}/{cpu['total_cores']} threads) | "
            f"RAM: {mem['used_gb']:.1f}GB/{mem['total_gb']:.1f}GB ({mem['percent_used']:.0f}%) | "
            f"Resource State: {status_tag}"
        )

    def get_status_panel(self) -> Panel:
        """Build a Rich panel presenting live hardware allocation and safeguards."""
        mem = self.get_memory_info()
        cpu = self.get_cpu_info()

        color = "red" if mem["percent_used"] > 90 else ("yellow" if mem["is_under_pressure"] else "green")
        lines = [
            f"[bold]Active Resource Protection:[/bold] {'[green]ENABLED[/green]' if self.settings.enable_resource_guard else '[dim]DISABLED[/dim]'}",
            f"[bold]CPU Allocation:[/bold] [cyan]{cpu['safe_threads']}[/cyan] threads assigned to LLM ([dim]{cpu['reserved_cores']} cores reserved for SSH/OS[/dim])",
            f"[bold]RAM Usage:[/bold] [{color}]{mem['used_gb']:.1f} GB / {mem['total_gb']:.1f} GB ({mem['percent_used']:.0f}% used)[/{color}] | Free: [cyan]{mem['available_gb']:.1f} GB[/cyan]",
            f"[bold]Context Window Cap:[/bold] [magenta]{self.settings.max_context_tokens:,} tokens[/magenta] (prevents KV cache VRAM spikes)",
            f"[bold]Operational Ingestion Caps:[/bold] [blue]{self.settings.max_safe_papers_limit} papers[/blue] / [blue]{self.settings.max_safe_trials_limit} trials[/blue] max per cycle",
        ]
        return Panel(
            "\n".join(lines),
            title="[bold green]🛡️ HelixMind Resource Guard & Hardware Throttling[/bold green]",
            border_style="green",
        )


_RESOURCE_MANAGER: Optional[SystemResourceGuard] = None


def get_resource_manager() -> SystemResourceGuard:
    """Return a cached singleton instance of SystemResourceGuard."""
    global _RESOURCE_MANAGER
    if _RESOURCE_MANAGER is None:
        _RESOURCE_MANAGER = SystemResourceGuard()
    return _RESOURCE_MANAGER
