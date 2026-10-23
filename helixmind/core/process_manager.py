"""Background Process Manager and Daemon Orchestrator for HelixMind.

Provides PM2-like process detachment, background mission persistence,
server auto-launch, live PID monitoring, log capture, and SSH-disconnect survival.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import httpx

from helixmind.config import get_settings

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_local_ip() -> str:
    """Determine the host machine's primary local LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not actually transmit packets, just resolves local routing interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class BackgroundMissionManager:
    """Manages background investigations and ensures API server availability."""

    def __init__(self, runs_dir: Optional[Path] = None):
        settings = get_settings()
        self.runs_dir = runs_dir or (settings.data_dir / "runs")
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def is_server_alive(self, host: str = "127.0.0.1", port: int = 8000) -> bool:
        """Check if HelixMind Web API is actively listening on host:port."""
        target_host = "127.0.0.1" if host == "0.0.0.0" else host
        url = f"http://{target_host}:{port}/api/health"
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(url)
                return res.status_code == 200
        except Exception:
            return False

    def ensure_server_running(self, host: str = "0.0.0.0", port: int = 8000) -> Dict[str, Any]:
        """Verify API server is up, or spawn it as a persistent background daemon."""
        if self.is_server_alive(host=host, port=port):
            local_ip = get_local_ip()
            return {
                "running": True,
                "already_active": True,
                "port": port,
                "local_url": f"http://localhost:{port}",
                "network_url": f"http://{local_ip}:{port}",
            }

        server_log_path = self.runs_dir / "server.log"
        server_log_file = open(server_log_path, "a", encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "helixmind.cli",
            "serve",
            "--host",
            host,
            "--port",
            str(port),
        ]

        # Spawn detached process immune to SSH SIGHUP
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # setsid() detaches from terminal
            close_fds=True,
        )

        # Wait up to 5 seconds for health endpoint
        for _ in range(25):
            time.sleep(0.2)
            if self.is_server_alive(host=host, port=port):
                break

        local_ip = get_local_ip()
        return {
            "running": True,
            "already_active": False,
            "pid": proc.pid,
            "port": port,
            "local_url": f"http://localhost:{port}",
            "network_url": f"http://{local_ip}:{port}",
            "log_file": str(server_log_path),
        }

    def start_mission(
        self,
        topic: str,
        depth: str = "deep",
        papers: Optional[int] = None,
        trials: Optional[int] = None,
        model: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_depmap: Optional[bool] = None,
        enable_pharmacovigilance: Optional[bool] = None,
        enable_clinvar: Optional[bool] = None,
        enable_tme: Optional[bool] = None,
        enable_crispr: Optional[bool] = None,
        enable_tribunal: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Launch an autonomous multi-agent scientific investigation as a detached background daemon."""
        # 1. Ensure Web Server is up so the user gets a live dashboard URL
        server_info = self.ensure_server_running(host=host, port=port)

        # 2. Build Mission Metadata
        inv_id = f"INV-{int(time.time())}"
        log_file_path = self.runs_dir / f"{inv_id}.log"
        meta_file_path = self.runs_dir / f"{inv_id}.json"

        local_ip = get_local_ip()
        dashboard_url = f"http://localhost:{port}?inv={inv_id}"
        network_url = f"http://{local_ip}:{port}?inv={inv_id}"

        # 3. Assemble CLI execution command
        cmd = [
            sys.executable,
            "-m",
            "helixmind.cli",
            "investigate",
            topic,
            "--depth",
            depth,
        ]
        if papers is not None:
            cmd.extend(["--papers", str(papers)])
        if trials is not None:
            cmd.extend(["--trials", str(trials)])
        if model:
            cmd.extend(["--model", model])
        if enable_depmap is False:
            cmd.append("--no-depmap")
        if enable_pharmacovigilance is False:
            cmd.append("--no-fda")
        if enable_clinvar is False:
            cmd.append("--no-clinvar")
        if enable_tme is False:
            cmd.append("--no-tme")
        if enable_crispr is False:
            cmd.append("--no-crispr")
        if enable_tribunal is False:
            cmd.append("--no-tribunal")

        # 4. Spawn detached daemon process
        log_out = open(log_file_path, "w", encoding="utf-8")
        env = dict(os.environ)
        env["HELIXMIND_INVESTIGATION_ID"] = inv_id
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_out,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # setsid() detaches from terminal/SSH session
            env=env,
            close_fds=True,
        )

        mission_meta = {
            "investigation_id": inv_id,
            "topic": topic,
            "pid": proc.pid,
            "status": "running",
            "depth": depth,
            "model": model or get_settings().ollama_default_model,
            "papers": papers or get_settings().default_literature_limit,
            "trials": trials or get_settings().default_trials_limit,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
            "log_file": str(log_file_path),
            "meta_file": str(meta_file_path),
            "dashboard_url": dashboard_url,
            "network_url": network_url,
        }

        with open(meta_file_path, "w", encoding="utf-8") as mf:
            json.dump(mission_meta, mf, indent=2)

        return mission_meta

    def list_missions(self) -> List[Dict[str, Any]]:
        """List all background missions, updating statuses with real-time process & hardware telemetry."""
        missions: List[Dict[str, Any]] = []
        meta_files = sorted(self.runs_dir.glob("INV-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        for mf in meta_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                pid = data.get("pid")
                current_status = data.get("status", "unknown")
                is_alive = False
                cpu_pct = 0.0
                mem_mb = 0.0

                if pid:
                    if HAS_PSUTIL:
                        try:
                            p = psutil.Process(pid)
                            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                                is_alive = True
                                cpu_pct = p.cpu_percent(interval=None)
                                mem_mb = round(p.memory_info().rss / (1024 * 1024), 1)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            is_alive = False
                    else:
                        try:
                            os.kill(pid, 0)
                            is_alive = True
                        except (OSError, ProcessLookupError):
                            is_alive = False

                # Check if report finished if process is no longer alive
                if not is_alive and current_status == "running":
                    # Check if report was generated in data/reports
                    inv_id = data.get("investigation_id", "")
                    reports_dir = get_settings().data_dir / "reports"
                    has_report = any(inv_id in r.name for r in reports_dir.glob("report_*.json"))
                    current_status = "completed" if has_report else "stopped"
                    data["status"] = current_status
                    with open(mf, "w", encoding="utf-8") as uf:
                        json.dump(data, uf, indent=2)

                # Elapsed time calculation
                start_ts = data.get("timestamp", 0)
                elapsed_sec = int(time.time() - start_ts) if start_ts else 0
                mins, secs = divmod(elapsed_sec, 60)
                hrs, mins = divmod(mins, 60)
                uptime_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s" if hrs else f"{mins:02d}m {secs:02d}s"

                missions.append({
                    **data,
                    "is_alive": is_alive,
                    "cpu_percent": cpu_pct,
                    "memory_mb": mem_mb,
                    "uptime": uptime_str,
                })
            except Exception:
                continue

        return missions

    def get_mission(self, inv_id_or_pid: str) -> Optional[Dict[str, Any]]:
        """Retrieve details for a specific mission by investigation ID or PID."""
        for m in self.list_missions():
            if m.get("investigation_id") == inv_id_or_pid or str(m.get("pid")) == str(inv_id_or_pid):
                return m
        return None

    def stop_mission(self, inv_id_or_pid: str) -> bool:
        """Stop a running background mission process."""
        target = self.get_mission(inv_id_or_pid)
        if not target or not target.get("pid"):
            return False

        pid = target["pid"]
        try:
            if HAS_PSUTIL:
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    p.wait(timeout=2.0)
                except psutil.TimeoutExpired:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1.0)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

            # Update meta file
            meta_path = Path(target.get("meta_file", self.runs_dir / f"{target['investigation_id']}.json"))
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["status"] = "stopped"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            return True
        except Exception:
            return False

    def get_logs(self, inv_id: str, lines: int = 100) -> str:
        """Read the recent console logs generated by a background mission."""
        log_file = self.runs_dir / f"{inv_id}.log"
        if not log_file.exists():
            return f"Log file for {inv_id} not found."

        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading logs: {e}"


def get_process_manager() -> BackgroundMissionManager:
    """Return singleton BackgroundMissionManager."""
    return BackgroundMissionManager()
