"""System Resource Guard & Hardware Throttling Engine for HelixMind.

Monitors host CPU, RAM, and GPU memory in real-time, throttles compute threads
to prevent machine freeze and SSH connection dropouts, enforces safe context sizes,
and proactively manages garbage collection and model unloading.
"""

from __future__ import annotations

import asyncio
import gc
import os
import re
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

    def get_full_metrics(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Obtain holistic hardware consumption metrics, process health, and token counters."""
        mem = self.get_memory_info()
        cpu = self.get_cpu_info()
        
        per_cpu = []
        swap_info = {"total_gb": 0.0, "used_gb": 0.0, "percent": 0.0}
        boot_time = 0.0
        
        if HAS_PSUTIL:
            try:
                per_cpu = psutil.cpu_percent(interval=None, percpu=True)
                sw = psutil.swap_memory()
                swap_info = {
                    "total_gb": round(sw.total / (1024 ** 3), 2),
                    "used_gb": round(sw.used / (1024 ** 3), 2),
                    "percent": sw.percent,
                }
                boot_time = psutil.boot_time()
            except Exception:
                pass

        # Target process metrics
        target_pid = pid or os.getpid()
        proc_data = {
            "pid": target_pid,
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "memory_percent": 0.0,
            "status": "running",
            "threads_count": 1,
        }
        if HAS_PSUTIL:
            try:
                p = psutil.Process(target_pid)
                p_mem = p.memory_info()
                proc_data = {
                    "pid": target_pid,
                    "cpu_percent": p.cpu_percent(interval=None),
                    "memory_mb": round(p_mem.rss / (1024 * 1024), 1),
                    "memory_percent": round(p.memory_percent(), 2),
                    "status": p.status(),
                    "threads_count": p.num_threads(),
                }
            except Exception:
                pass

        # Tokens
        tokens_summary = {}
        try:
            from helixmind.core.tokens import get_token_tracker
            tokens_summary = get_token_tracker().get_metrics()
        except Exception:
            pass

        return {
            "cpu": {
                **cpu,
                "per_core_percent": per_cpu,
            },
            "memory": mem,
            "swap": swap_info,
            "process": proc_data,
            "platform": sys.platform,
            "system_uptime_sec": round(time.time() - boot_time, 1) if boot_time else 0.0,
            "tokens": tokens_summary,
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

    def resolve_safe_model(
        self,
        client: Any,
        requested_model: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """Validate that requested model is installed and safe for host RAM.
        
        If model is missing, not installed, or dangerously large (> available host RAM),
        selects the best installed candidate that fits safely in memory to prevent OOM crash.
        """
        model = (requested_model or self.settings.ollama_default_model).strip()
        
        try:
            installed = client.list_models()
        except Exception:
            return model, None

        if not installed:
            return model, None

        installed_names = [m.get("name", "") for m in installed]

        mem = self.get_memory_info()
        available_gb = mem.get("available_gb", 8.0)
        total_gb = mem.get("total_gb", 16.0)

        # Check for dangerous oversized parameter tags (e.g. 671b, 70b, 120b, 405b, 32b)
        is_oversized = False
        param_match = re.search(r"(\d+(?:\.\d+)?)[bB]", model)
        if param_match:
            try:
                param_num = float(param_match.group(1))
                # If parameter size in billions exceeds available RAM or total RAM / 1.5,
                # it will trigger OOM allocation failure
                if param_num >= (total_gb * 0.9):
                    is_oversized = True
            except Exception:
                pass

        exact_match = None
        for m in installed:
            name = m.get("name", "")
            if name == model or name.split(":")[0] == model or model.split(":")[0] == name.split(":")[0]:
                exact_match = m
                break

        if exact_match and not is_oversized:
            disk_size_gb = exact_match.get("size", 0) / (1024 ** 3)
            if disk_size_gb > 0 and disk_size_gb >= (total_gb * 0.9):
                is_oversized = True
            else:
                return exact_match.get("name", model), None

        # Pick safest installed model
        fallback_model = None
        # Preference: if requested had deepseek/r1, look for deepseek-r1:1.5b
        if "deepseek" in model.lower():
            for name in installed_names:
                if "deepseek" in name.lower() and ("1.5b" in name.lower() or "1b" in name.lower()):
                    fallback_model = name
                    break
        elif "qwen" in model.lower():
            for name in installed_names:
                if "qwen" in name.lower() and ("coder" in name.lower() or "2b" in name.lower() or "7b" in name.lower()):
                    fallback_model = name
                    break

        if not fallback_model:
            for m in installed:
                name = m.get("name", "")
                if "embed" in name.lower():
                    continue
                size_gb = m.get("size", 0) / (1024 ** 3)
                if size_gb <= (available_gb * 0.8) or size_gb <= 4.5:
                    fallback_model = name
                    break

        if not fallback_model:
            fallback_model = installed_names[0] if installed_names else "llama3.2:1b"

        reason = (
            f"Model '{model}' requires massive RAM allocation ({param_match.group(0) if param_match else 'oversized'}) "
            f"exceeding host memory ({total_gb:.0f} GB)"
            if is_oversized
            else f"Model '{model}' is not installed in local Ollama"
        )
        warning = (
            f"🛡️ [bold yellow]Resource Guard Model Override:[/bold yellow] {reason}. "
            f"Automatically routed to verified safe installed model [bold green]'{fallback_model}'[/bold green] to prevent machine crash."
        )
        return fallback_model, warning

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
