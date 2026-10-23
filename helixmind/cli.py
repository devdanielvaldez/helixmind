"""HelixMind Command Line Interface.

Provides an interactive console menu, multi-agent scientific investigation engine,
AI model configuration, research parameter tuning, standalone in-silico tools,
and system diagnostics.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from helixmind import __version__
from helixmind.api import start_server
from helixmind.autopilot import AutoPilotOrchestrator
from helixmind.config import get_settings, save_env_variables, update_settings
from helixmind.copilot import ResearchCopilot
from helixmind.core.process_manager import get_process_manager
from helixmind.hypotheses.models import ResearchReport
from helixmind.insilico import (
    ClinicalSimulator,
    ClinVarStratifier,
    CRISPRDesigner,
    DeNovoGenerator,
    DepMapVerifier,
    DockingSimulator,
    FDAPharmacovigilance,
    TMEMapper,
)
from helixmind.llm.ollama_client import OllamaClient, OllamaConnectionError

console = Console()

BANNER_ART = r"""
  _    _      _ _      __  __ _           _ 
 | |  | |    | (_)    |  \/  (_)         | |
 | |__| | ___| |___  _| \  / |_ _ __   __| |
 |  __  |/ _ \ | \ \/ / |\/| | | '_ \ / _` |
 | |  | |  __/ | |>  <| |  | | | | | | (_| |
 |_|  |_|\___|_|_/_/\_\_|  |_|_|_| |_|\__,_|
"""


# =============================================================================
# Diagnostic & Information Helpers
# =============================================================================


def display_banner() -> None:
    """Print the primary HelixMind ASCII banner and version header."""
    console.print(f"[bold cyan]{BANNER_ART}[/bold cyan]")
    console.print(
        f"[bold white]Autonomous AI Scientific Research Platform[/bold white] "
        f"[dim]v{__version__}[/dim]\n"
    )


def get_system_status_panel(client: OllamaClient) -> Panel:
    """Build a rich status summary panel of current platform configuration."""
    settings = get_settings()
    is_alive = client.check_health()
    status_str = "[bold green]ONLINE[/bold green]" if is_alive else "[bold red]OFFLINE (Start with: ollama serve)[/bold red]"

    modules_status = []
    modules_status.append(f"DepMap: {'[green]ON[/green]' if settings.enable_depmap else '[dim]OFF[/dim]'}")
    modules_status.append(f"openFDA: {'[green]ON[/green]' if settings.enable_pharmacovigilance else '[dim]OFF[/dim]'}")
    modules_status.append(f"ClinVar: {'[green]ON[/green]' if settings.enable_clinvar else '[dim]OFF[/dim]'}")
    modules_status.append(f"TME: {'[green]ON[/green]' if settings.enable_tme else '[dim]OFF[/dim]'}")
    modules_status.append(f"CRISPR: {'[green]ON[/green]' if settings.enable_crispr else '[dim]OFF[/dim]'}")
    modules_status.append(f"Tribunal: {'[green]ON[/green]' if settings.enable_tribunal else '[dim]OFF[/dim]'}")

    from helixmind.core.resources import get_resource_manager
    rm = get_resource_manager()
    mem = rm.get_memory_info()
    cpu = rm.get_cpu_info()
    resource_line = (
        f"[bold]Resource Guard:[/bold] {'[green]ACTIVE[/green]' if settings.enable_resource_guard else '[dim]OFF[/dim]'} | "
        f"CPU: [cyan]{cpu['safe_threads']}/{cpu['total_cores']} threads[/cyan] ([dim]{cpu['reserved_cores']} reserved for SSH/OS[/dim]) | "
        f"RAM: [cyan]{mem['used_gb']:.1f}GB/{mem['total_gb']:.1f}GB ({mem['percent_used']:.0f}%)[/cyan]"
    )

    text = (
        f"[bold]Ollama Server:[/bold] {status_str} [dim]({client.base_url})[/dim]\n"
        f"[bold]Active Models:[/bold] Default: [cyan]{settings.ollama_default_model}[/cyan] | "
        f"Reasoning: [yellow]{settings.ollama_reasoning_model}[/yellow] | "
        f"Embedding: [blue]{settings.ollama_embedding_model}[/blue]\n"
        f"{resource_line}\n"
        f"[bold]Defaults:[/bold] Depth: [magenta]{settings.default_depth.upper()}[/magenta] | "
        f"Papers Limit: [cyan]{settings.default_literature_limit}[/cyan] (Max safe: [dim]{settings.max_safe_papers_limit}[/dim]) | "
        f"Trials Limit: [cyan]{settings.default_trials_limit}[/cyan] (Max safe: [dim]{settings.max_safe_trials_limit}[/dim])\n"
        f"[bold]Pipeline Modules:[/bold] {' | '.join(modules_status)}\n"
        f"[bold]Export Formats:[/bold] [dim]{settings.export_formats.upper()}[/dim]"
    )
    return Panel(text, title="[bold cyan]HelixMind System Configuration[/bold cyan]", border_style="cyan")


def fetch_installed_models(client: OllamaClient) -> List[Dict[str, Any]]:
    """Return list of models installed in Ollama or empty list if offline."""
    try:
        return client.list_models()
    except Exception:
        return []


# =============================================================================
# Command Implementations
# =============================================================================


def cmd_status(client: OllamaClient) -> None:
    """Check and display Ollama service status and available models."""
    settings = get_settings()
    console.print(
        Panel.fit(
            f"[bold cyan]HelixMind v{__version__}[/bold cyan] - Scientific Research Platform\n"
            f"Ollama Endpoint: [green]{client.base_url}[/green]",
            title="System Diagnostics",
            border_style="cyan",
        )
    )

    console.print("Checking Ollama server connection...")
    is_alive = client.check_health()
    if not is_alive:
        console.print(
            f"[bold red]✗ Failed to connect to Ollama at {client.base_url}[/bold red]\n"
            "Ensure Ollama is running (`ollama serve`)."
        )
        return

    console.print("[bold green]✓ Ollama server is active and reachable![/bold green]\n")
    models = fetch_installed_models(client)

    if not models:
        console.print("[yellow]No models found in Ollama. Pull one with `ollama pull llama3.2:1b`.[/yellow]")
        return

    table = Table(title="Installed Ollama Models", header_style="bold magenta")
    table.add_column("Model Name", style="cyan")
    table.add_column("Parameter Size", justify="right")
    table.add_column("Quantization", justify="center")
    table.add_column("Disk Size", justify="right")
    table.add_column("Role / Status", style="green")

    for m in models:
        name = m.get("name", "unknown")
        details = m.get("details", {})
        param_size = details.get("parameter_size", "N/A")
        quant = details.get("quantization_level", "N/A")
        size_bytes = m.get("size", 0)
        size_gb = f"{size_bytes / (1024 ** 3):.2f} GB"

        role = []
        if name == settings.ollama_default_model or name.split(":")[0] == settings.ollama_default_model:
            role.append("[bold green]DEFAULT[/bold green]")
        if name == settings.ollama_reasoning_model or name.split(":")[0] == settings.ollama_reasoning_model:
            role.append("[bold yellow]REASONING[/bold yellow]")
        if name == settings.ollama_embedding_model or name.split(":")[0] == settings.ollama_embedding_model:
            role.append("[bold blue]EMBEDDINGS[/bold blue]")

        table.add_row(name, param_size, quant, size_gb, ", ".join(role) if role else "[dim]available[/dim]")

    console.print(table)


def cmd_test_chat(client: OllamaClient, prompt: str, model: str | None = None) -> None:
    """Send a test prompt and stream the response to console."""
    settings = get_settings()
    selected_model = model or settings.ollama_default_model
    console.print(f"\n[cyan]Sending prompt to model [bold]{selected_model}[/bold]...[/cyan]")
    console.print(f"[dim]Prompt: {prompt}[/dim]\n")

    messages = [
        {"role": "system", "content": "You are HelixMind, an expert biomedical research assistant."},
        {"role": "user", "content": prompt},
    ]

    console.print("[bold green]Response:[/bold green]")
    try:
        for chunk in client.chat_stream(messages=messages, model=selected_model):
            console.print(chunk, end="")
        console.print("\n")
    except Exception as e:
        console.print(f"\n[bold red]Error during chat:[/bold red] {e}")


def cmd_test_embed(client: OllamaClient, sample_text: str, model: str | None = None) -> None:
    """Generate and display embeddings vector info."""
    settings = get_settings()
    selected_model = model or settings.ollama_embedding_model
    console.print(f"\n[cyan]Generating embeddings with model [bold]{selected_model}[/bold]...[/cyan]")
    console.print(f"[dim]Input: '{sample_text}'[/dim]")

    try:
        embeddings = client.embed(texts=sample_text, model=selected_model)
        if embeddings and len(embeddings) > 0:
            vec = embeddings[0]
            console.print("[bold green]✓ Embedding generated successfully![/bold green]")
            console.print(f"Dimension: [bold]{len(vec)}[/bold] floats")
            preview = ", ".join(f"{v:.4f}" for v in vec[:5]) + f", ... (total {len(vec)})"
            console.print(f"Vector preview: [dim][{preview}][/dim]\n")
        else:
            console.print("[bold red]✗ No embedding returned.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during embedding generation:[/bold red] {e}")


def cmd_investigate(
    topic: str,
    max_papers: int | None = None,
    max_trials: int | None = None,
    depth: str | None = None,
    model: str | None = None,
    enable_depmap: bool | None = None,
    enable_pharmacovigilance: bool | None = None,
    enable_clinvar: bool | None = None,
    enable_tme: bool | None = None,
    enable_crispr: bool | None = None,
    enable_tribunal: bool | None = None,
    export_formats: List[str] | None = None,
) -> ResearchReport:
    """Launch an autonomous multi-agent scientific investigation."""
    import signal
    if hasattr(signal, "SIGHUP"):
        try:
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except Exception:
            pass

    settings = get_settings()
    actual_papers = max_papers if max_papers is not None else settings.default_literature_limit
    actual_trials = max_trials if max_trials is not None else settings.default_trials_limit
    actual_depth = depth or settings.default_depth
    actual_model = model or settings.ollama_default_model

    actual_depmap = enable_depmap if enable_depmap is not None else settings.enable_depmap
    actual_pv = enable_pharmacovigilance if enable_pharmacovigilance is not None else settings.enable_pharmacovigilance
    actual_clinvar = enable_clinvar if enable_clinvar is not None else settings.enable_clinvar
    actual_tme = enable_tme if enable_tme is not None else settings.enable_tme
    actual_crispr = enable_crispr if enable_crispr is not None else settings.enable_crispr
    actual_tribunal = enable_tribunal if enable_tribunal is not None else settings.enable_tribunal
    actual_exports = export_formats if export_formats is not None else [f.strip() for f in settings.export_formats.split(",") if f.strip()]

    orchestrator = AutoPilotOrchestrator()
    console.print(
        Panel.fit(
            f"[bold green]Starting HelixMind Autonomous Investigation[/bold green]\n"
            f"Topic: [bold yellow]{topic}[/bold yellow]\n"
            f"Depth Level: [bold cyan]{actual_depth.upper()}[/bold cyan] | Target Papers: [cyan]{actual_papers}[/cyan] | Clinical Trials: [cyan]{actual_trials}[/cyan]\n"
            f"Active Model: [magenta]{actual_model}[/magenta] | Exports: [blue]{', '.join(actual_exports)}[/blue]",
            title="Auto-Pilot Research Engine",
            border_style="green",
        )
    )

    report: ResearchReport = asyncio.run(
        orchestrator.investigate_async(
            topic=topic,
            max_papers=actual_papers,
            max_trials=actual_trials,
            depth=actual_depth,
            model=actual_model,
            enable_depmap=actual_depmap,
            enable_pharmacovigilance=actual_pv,
            enable_clinvar=actual_clinvar,
            enable_tme=actual_tme,
            enable_crispr=actual_crispr,
            enable_tribunal=actual_tribunal,
            export_formats=actual_exports,
        )
    )

    # Render Report Summary
    console.rule("[bold green]Investigation Findings & Hypotheses[/bold green]")
    console.print(
        f"\n[bold]Analyzed:[/bold] {report.papers_analyzed} papers | {report.clinical_trials_analyzed} trials\n"
        f"[bold]Discovered Targets:[/bold] {', '.join(report.entities_discovered[:10])}\n"
    )

    for i, hyp in enumerate(report.hypotheses, 1):
        verdict_color = "green" if "Strongly" in (hyp.critique_verdict or "") or "Plausible" in (hyp.critique_verdict or "") else "yellow"
        hyp_content = (
            f"[bold cyan]Targets:[/bold cyan] {', '.join(hyp.biological_targets)}\n\n"
            f"[bold]Mechanism of Action:[/bold]\n{hyp.mechanism_of_action}\n\n"
            f"[bold]Testable Predictions:[/bold]\n" + "\n".join([f"• {p}" for p in hyp.testable_predictions]) + "\n\n"
            f"[bold]Validation Strategy:[/bold] {hyp.validation_strategy}\n"
            f"[bold]Confidence Score:[/bold] {hyp.confidence_score * 100:.0f}%\n"
            f"[bold]Reviewer Verdict:[/bold] [{verdict_color}]{hyp.critique_verdict or 'Pending'}[/{verdict_color}]\n"
        )
        if hyp.counter_arguments:
            hyp_content += f"[bold red]Caveats / Risks:[/bold red] {'; '.join(hyp.counter_arguments[:2])}\n"

        console.print(Panel(hyp_content, title=f"Hypothesis {i}: {hyp.title}", border_style="cyan"))

    if report.next_research_milestones:
        console.rule("[bold cyan]Recommended Experimental Roadmap[/bold cyan]")
        for m in report.next_research_milestones:
            console.print(f"  [bold green]→[/bold green] {m}")

    if report.paper_markdown_path:
        console.print(
            Panel.fit(
                f"[bold green]✓ Full Publication-Ready Scientific Paper Authored![/bold green]\n"
                f"File: [bold cyan]{report.paper_markdown_path}[/bold cyan]\n"
                f"Format: [white]GitHub Markdown, Figures & Cross-Referenced Literature[/white]",
                title="Scientific Paper Output",
                border_style="magenta",
            )
        )

    return report


def cmd_start_daemon(
    topic: str,
    papers: Optional[int] = None,
    trials: Optional[int] = None,
    depth: Optional[str] = None,
    model: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_depmap: Optional[bool] = None,
    enable_pharmacovigilance: Optional[bool] = None,
    enable_clinvar: Optional[bool] = None,
    enable_tme: Optional[bool] = None,
    enable_crispr: Optional[bool] = None,
    enable_tribunal: Optional[bool] = None,
) -> None:
    """Launch an autonomous multi-agent investigation detached in background (PM2-style)."""
    pm = get_process_manager()
    mission = pm.start_mission(
        topic=topic,
        depth=depth or "deep",
        papers=papers,
        trials=trials,
        model=model,
        host=host,
        port=port,
        enable_depmap=enable_depmap,
        enable_pharmacovigilance=enable_pharmacovigilance,
        enable_clinvar=enable_clinvar,
        enable_tme=enable_tme,
        enable_crispr=enable_crispr,
        enable_tribunal=enable_tribunal,
    )

    inv_id = mission["investigation_id"]
    pid = mission["pid"]
    local_url = mission["dashboard_url"]
    net_url = mission["network_url"]
    log_file = mission["log_file"]

    card = (
        f"[bold white]Topic:[/bold white] [bold yellow]{topic}[/bold yellow]\n"
        f"[bold white]Mission ID:[/bold white] [bold cyan]{inv_id}[/bold cyan]  |  [bold white]Daemon PID:[/bold white] [bold magenta]{pid}[/bold magenta]\n"
        f"[bold white]Depth:[/bold white] [cyan]{mission['depth'].upper()}[/cyan]  |  [bold white]Model:[/bold white] [magenta]{mission['model']}[/magenta]\n\n"
        f"[bold green]🌐 Live Interactive War Room Dashboard URL:[/bold green]\n"
        f"  • [bold]Local Host:[/bold]   [bold cyan]{local_url}[/bold cyan]\n"
        f"  • [bold]Remote / LAN:[/bold] [bold cyan]{net_url}[/bold cyan]\n\n"
        f"[bold green]🛡️ SSH SAFE & PM2-EQUIVALENT DETACHED:[/bold green]\n"
        f"  You can now [bold red]safely close this terminal / SSH connection[/bold red].\n"
        f"  The investigation will continue running uninhibited in background.\n\n"
        f"[bold]Useful CLI Controls:[/bold]\n"
        f"  • [cyan]helixmind ps[/cyan]               List active background tasks & live CPU/RAM/Tokens\n"
        f"  • [cyan]helixmind logs {inv_id} -f[/cyan]    Stream live terminal logs from background\n"
        f"  • [cyan]helixmind stop {inv_id}[/cyan]      Stop this background investigation\n"
        f"  • [dim]tail -f {log_file}[/dim]\n"
    )
    console.print(Panel(card, title="🚀 HelixMind Autonomous Mission Started (Background)", border_style="green"))


def cmd_ps() -> None:
    """List all background missions with live PID, hardware load, and token metrics."""
    pm = get_process_manager()
    missions = pm.list_missions()

    if not missions:
        console.print("[dim]No background missions found. Start one with: [bold cyan]helixmind start \"topic\"[/bold cyan][/dim]")
        return

    table = Table(title="HelixMind Background Missions & Telemetry", border_style="cyan")
    table.add_column("Mission ID", style="cyan", no_wrap=True)
    table.add_column("Topic", style="yellow")
    table.add_column("PID", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("CPU %", style="green")
    table.add_column("RAM", style="blue")
    table.add_column("Uptime", style="dim")
    table.add_column("Dashboard URL", style="cyan")

    for m in missions:
        st = m.get("status", "unknown").upper()
        if st == "RUNNING":
            status_style = "[bold green]RUNNING[/bold green]"
        elif st == "COMPLETED":
            status_style = "[bold cyan]COMPLETED[/bold cyan]"
        elif st == "STOPPED":
            status_style = "[bold yellow]STOPPED[/bold yellow]"
        else:
            status_style = f"[dim]{st}[/dim]"

        cpu_str = f"{m.get('cpu_percent', 0.0):.1f}%" if m.get("is_alive") else "-"
        ram_str = f"{m.get('memory_mb', 0.0):.1f} MB" if m.get("is_alive") else "-"

        table.add_row(
            m.get("investigation_id", "-"),
            m.get("topic", "-")[:35],
            str(m.get("pid", "-")),
            status_style,
            cpu_str,
            ram_str,
            m.get("uptime", "-"),
            m.get("dashboard_url", "-"),
        )

    console.print(table)


def cmd_logs(inv_id_or_pid: str, follow: bool = False, lines: int = 100) -> None:
    """View or stream live logs from a background investigation."""
    pm = get_process_manager()
    mission = pm.get_mission(inv_id_or_pid)

    log_path = None
    if mission and mission.get("log_file"):
        log_path = Path(mission["log_file"])
    else:
        for candidate in (pm.runs_dir / f"{inv_id_or_pid}.log", pm.runs_dir / f"INV-{inv_id_or_pid}.log"):
            if candidate.exists():
                log_path = candidate
                break

    if not log_path or not log_path.exists():
        console.print(f"[bold red]Log file for '{inv_id_or_pid}' not found.[/bold red]")
        return

    if not follow:
        content = pm.get_logs(inv_id_or_pid, lines=lines)
        console.print(content)
        return

    console.print(f"[dim]Streaming logs from {log_path} (Press Ctrl+C to detach)...[/dim]\n")
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            # Print initial chunk
            f.seek(0, os.SEEK_END)
            end_pos = f.tell()
            f.seek(max(0, end_pos - (lines * 120)))
            for line in f:
                sys.stdout.write(line)
                sys.stdout.flush()
            # Follow live
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.3)
    except KeyboardInterrupt:
        console.print("\n[dim]Detached from log stream. Background task is still running uninhibited.[/dim]")


def cmd_stop(inv_id_or_pid: str) -> None:
    """Stop a running background mission."""
    pm = get_process_manager()
    success = pm.stop_mission(inv_id_or_pid)
    if success:
        console.print(f"[bold green]✓ Successfully stopped background mission: {inv_id_or_pid}[/bold green]")
    else:
        console.print(f"[bold red]Failed to stop mission '{inv_id_or_pid}'. Process not running or not found.[/bold red]")


# =============================================================================
# Standalone In-Silico Molecular & Genomic Suite
# =============================================================================


def run_tool_depmap(gene: str, condition: str = "Oncology") -> None:
    """Run DepMap cancer dependency verification for a specific target."""
    console.print(f"\n[cyan]Querying Broad Institute DepMap functional genomics for [bold]{gene}[/bold]...[/cyan]")
    verifier = DepMapVerifier()
    res = asyncio.run(verifier.verify_gene_essentiality_async(gene, target_condition=condition))
    tier_color = "red" if "ESSENTIAL" in res.essentiality_tier else "cyan"

    table = Table(title=f"DepMap Chronos Dependency: {gene}", header_style="bold magenta")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Gene Symbol", res.gene_symbol)
    table.add_row("Chronos Score", f"[{tier_color}]{res.chronos_score}[/{tier_color}]")
    table.add_row("Essentiality Tier", f"[{tier_color}]{res.essentiality_tier}[/{tier_color}]")
    table.add_row("Cell Lines Tested", str(res.cell_lines_tested_count))
    table.add_row("Selective in Lineage", res.selective_in_lineage)
    table.add_row("Actionability", res.therapeutic_actionability)

    console.print(table)


def run_tool_openfda(drug: str) -> None:
    """Run openFDA FAERS adverse event surveillance for a given therapeutic."""
    console.print(f"\n[cyan]Querying openFDA FAERS post-marketing adverse reactions for [bold]{drug}[/bold]...[/cyan]")
    fda = FDAPharmacovigilance()
    res = asyncio.run(fda.fetch_safety_profile_async(drug, limit=8))

    console.print(
        Panel.fit(
            f"[bold]Drug Name:[/bold] {res.drug_name}\n"
            f"[bold]Total FAERS Patient Reports:[/bold] {res.total_patient_reports:,}\n"
            f"[bold]Clinical Safety Verdict:[/bold] [green]{res.clinical_safety_verdict}[/green]",
            title="FDA Clinical Pharmacovigilance",
            border_style="blue",
        )
    )

    table = Table(title="Top Reported MedDRA Adverse Events", header_style="bold cyan")
    table.add_column("Adverse Event (MedDRA PT)", style="bold")
    table.add_column("Report Count", justify="right")
    table.add_column("Incidence %", justify="right")
    table.add_column("Severity Classification", style="magenta")

    for r in res.top_adverse_reactions:
        table.add_row(r.meddra_term, f"{r.report_count:,}", f"{r.percentage_of_reports:.1f}%", r.severity_tier)

    console.print(table)


def run_tool_clinvar(gene: str, condition: str = "Oncology") -> None:
    """Run NCBI ClinVar genomic patient stratification for a target gene."""
    console.print(f"\n[cyan]Querying NCBI ClinVar pathogenic variants for [bold]{gene}[/bold] in {condition}...[/cyan]")
    stratifier = ClinVarStratifier()
    res = asyncio.run(stratifier.stratify_target_async(gene, target_condition=condition, max_variants=5))

    console.print(
        Panel.fit(
            f"[bold]Target Gene:[/bold] {res.target_gene} | [bold]Condition:[/bold] {res.target_condition}\n"
            f"[bold]Verified Pathogenic Variants in ClinVar:[/bold] {res.total_pathogenic_variants_in_db}\n"
            f"[bold]Responder Biomarker Signature:[/bold] {res.responder_biomarker_signature}\n"
            f"[bold]Refractory Gatekeeper Signature:[/bold] {res.resistance_gatekeeper_signature}\n"
            f"[bold]Clinical Trial NGS Protocol:[/bold] {res.clinical_trial_genomic_criteria}",
            title="NCBI ClinVar Genomic Patient Stratification",
            border_style="magenta",
        )
    )

    if res.top_pathogenic_variants:
        table = Table(title="Representative Pathogenic Variants", header_style="bold green")
        table.add_column("Variant HGVS", style="bold cyan")
        table.add_column("Clinical Significance", style="yellow")
        table.add_column("Consequence", style="white")
        table.add_column("ClinVar Review Status", style="dim")

        for v in res.top_pathogenic_variants:
            table.add_row(v.title[:45], v.clinical_significance, v.molecular_consequence, v.review_status)
        console.print(table)


def run_tool_tme(genes_str: str, condition: str = "Solid Tumor") -> None:
    """Run single-cell tumor microenvironment and paracrine interactome deconvolution."""
    genes = [g.strip().upper() for g in genes_str.split(",") if g.strip()]
    if not genes:
        genes = ["STAT3", "CD274", "TGFB1"]

    console.print(f"\n[cyan]Mapping Single-Cell Tumor Microenvironment for {', '.join(genes)} in {condition}...[/cyan]")
    res = TMEMapper.analyze_microenvironment(genes, condition=condition)

    console.print(
        Panel.fit(
            f"[bold]Condition:[/bold] {res.target_disease}\n"
            f"[bold]Microenvironment Subtype:[/bold] [magenta]{res.microenvironment_subtype}[/magenta]\n"
            f"[bold]Stromal Immunosuppression Index:[/bold] [red]{res.stromal_immunosuppression_index:.1f}/100[/red]\n"
            f"[bold]Recommended Therapeutic Attack:[/bold] {res.combination_strategy}",
            title="Single-Cell TME Interactome",
            border_style="magenta",
        )
    )

    if res.active_ligand_receptor_axes:
        table = Table(title="Active Paracrine Cell-Cell Axes", header_style="bold yellow")
        table.add_column("Source Cell", style="cyan")
        table.add_column("Ligand - Receptor Axis", style="bold green")
        table.add_column("Target Cell", style="magenta")
        table.add_column("Functional Effect", style="white")

        for ax in res.active_ligand_receptor_axes:
            table.add_row(ax.source_cell_type, f"{ax.ligand_gene} ➔ {ax.receptor_gene}", ax.target_cell_type, ax.functional_effect)
        console.print(table)


def run_tool_crispr(gene: str) -> None:
    """Design CRISPR sgRNAs and golden gate validation primers for a target gene."""
    console.print(f"\n[cyan]Synthesizing CRISPR-Cas9 sgRNAs and validation primers for [bold]{gene}[/bold]...[/cyan]")
    designer = CRISPRDesigner()
    res = asyncio.run(designer.design_for_gene_async(gene_name=gene))

    console.print(
        Panel.fit(
            f"[bold]Target Gene:[/bold] {res.target_gene} | [bold]Organism:[/bold] {res.organism}\n"
            f"[bold]Top Guide Sequence (5'->3'):[/bold] [green]{res.top_guide_sequence}[/green] [dim](PAM: {res.pam_site})[/dim]\n"
            f"[bold]Predicted Cleavage Efficiency Score:[/bold] [cyan]{res.on_target_score}/100[/cyan]",
            title="CRISPR sgRNA Laboratory Knockout Guide",
            border_style="green",
        )
    )

    if res.top_sgrnas:
        table = Table(title="Candidate sgRNAs & Golden Gate Cloning Oligos", header_style="bold cyan")
        table.add_column("#", justify="center")
        table.add_column("Guide (20-nt)", style="bold")
        table.add_column("PAM", style="yellow")
        table.add_column("GC %", justify="right")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Forward Cloning Oligo (BsmBI)", style="dim")

        for idx, g in enumerate(res.top_sgrnas[:4], 1):
            table.add_row(str(idx), g.sequence_20nt, g.pam_motif, f"{g.gc_percent:.1f}%", f"{g.on_target_score:.0f}", g.oligo_forward_top)
        console.print(table)

    if res.validation_primers:
        p = res.validation_primers
        console.print(
            Panel.fit(
                f"[bold]Forward PCR Primer:[/bold] {p.forward_primer} (Tm: {p.forward_tm} °C, GC: {p.gc_percent_forward:.1f}%)\n"
                f"[bold]Reverse PCR Primer:[/bold] {p.reverse_primer} (Tm: {p.reverse_tm} °C)\n"
                f"[bold]Expected Genomic Amplicon Length:[/bold] {p.amplicon_length_bp} bp",
                title="Validation PCR Primers",
                border_style="blue",
            )
        )


def run_tool_denovo(parent_drug: str) -> None:
    """Generate de novo bioisosteric chemical analogs and compute MMFF94 force field affinity."""
    console.print(f"\n[cyan]Generating de novo bioisosteric chemical analogs from [bold]{parent_drug}[/bold]...[/cyan]")
    analogs = DeNovoGenerator.generate_optimized_analogs(parent_drug, max_analogs=3)
    if not analogs:
        console.print("[yellow]Could not generate analogs for specified compound.[/yellow]")
        return

    table = Table(title=f"De Novo Optimized Candidates Derived from {parent_drug.title()}", header_style="bold magenta")
    table.add_column("Code", style="bold cyan")
    table.add_column("Rationale / Chemical Innovation", style="white")
    table.add_column("SAscore (1-10)", justify="right", style="green")
    table.add_column("Synthesizability", style="magenta")
    table.add_column("Predicted BBB", justify="center")
    table.add_column("hERG Cardiac Risk", justify="center")

    for a in analogs:
        table.add_row(
            a.compound_code,
            a.modification_rationale[:60] + "...",
            f"{a.sa_score:.2f}",
            a.sa_verdict,
            "[green]YES[/green]" if a.bbb_penetrant else "[dim]NO[/dim]",
            f"[{'red' if a.herg_cardiotox_risk == 'HIGH' else ('yellow' if a.herg_cardiotox_risk == 'MEDIUM' else 'green')}]{a.herg_cardiotox_risk}[/]",
        )
    console.print(table)


def run_tool_clinical_simulation(drug: str, condition: str = "Metastatic Oncology") -> None:
    """Execute virtual patient twin clinical trial simulation and Kaplan-Meier estimation."""
    console.print(f"\n[cyan]Simulating virtual Phase 2b trial: [bold]{drug}[/bold] vs Standard-of-Care in {condition}...[/cyan]")
    res = ClinicalSimulator.simulate_trial(drug, target_condition=condition, cohort_size=200)

    console.print(
        Panel.fit(
            f"[bold]Trial Protocol:[/bold] {res.trial_title}\n"
            f"[bold]Virtual Patient Cohort:[/bold] {res.total_patients_simulated} digital twin patients ({res.virtual_cohort_size_per_arm} per arm)\n"
            f"[bold]Hazard Ratio (HR):[/bold] [bold green]{res.hazard_ratio:.2f}[/bold green]\n"
            f"[bold]Log-Rank p-value:[/bold] [cyan]{res.logrank_p_value:.4f}[/cyan]\n"
            f"[bold]Median Progression-Free Survival (PFS):[/bold] Exp: [green]{res.median_pfs_experimental_months:.1f} mo[/green] vs Control: {res.median_pfs_control_months:.1f} mo [bold cyan](+{res.pfs_gain_months:.1f} mo)[/bold cyan]\n"
            f"[bold]Probability of Technical Success (PoS):[/bold] [bold cyan]{res.probability_of_technical_success:.1f}%[/bold cyan]\n"
            f"[bold]Clinical Outcome Verdict:[/bold] {res.trial_verdict}",
            title="In-Silico Virtual Clinical Trial Simulation",
            border_style="green",
        )
    )


# =============================================================================
# Interactive Submenus
# =============================================================================


def menu_configure_models(client: OllamaClient) -> None:
    """Interactive submenu for viewing, selecting, and pulling AI models."""
    while True:
        settings = get_settings()
        models = fetch_installed_models(client)

        console.rule("[bold cyan]Configure AI Models & LLM Providers[/bold cyan]")
        console.print(
            f"[bold]Ollama Server:[/bold] [green]{client.base_url}[/green]\n"
            f"[bold]Default Agent LLM:[/bold] [cyan]{settings.ollama_default_model}[/cyan]\n"
            f"[bold]Reasoning / Thinking LLM:[/bold] [yellow]{settings.ollama_reasoning_model}[/yellow]\n"
            f"[bold]Embedding Model:[/bold] [blue]{settings.ollama_embedding_model}[/blue]\n"
        )

        if models:
            table = Table(title="Installed Models in Ollama", header_style="bold magenta")
            table.add_column("#", justify="center", style="bold")
            table.add_column("Model Tag", style="cyan")
            table.add_column("Params", justify="right")
            table.add_column("Quant", justify="center")
            table.add_column("Size", justify="right")
            table.add_column("Assigned Role", style="green")

            for idx, m in enumerate(models, 1):
                name = m.get("name", "unknown")
                details = m.get("details", {})
                param = details.get("parameter_size", "N/A")
                quant = details.get("quantization_level", "N/A")
                size_gb = f"{m.get('size', 0) / (1024 ** 3):.2f} GB"

                roles = []
                if name == settings.ollama_default_model or name.split(":")[0] == settings.ollama_default_model:
                    roles.append("[bold green]DEFAULT[/bold green]")
                if name == settings.ollama_reasoning_model or name.split(":")[0] == settings.ollama_reasoning_model:
                    roles.append("[bold yellow]REASONING[/bold yellow]")
                if name == settings.ollama_embedding_model or name.split(":")[0] == settings.ollama_embedding_model:
                    roles.append("[bold blue]EMBEDDING[/bold blue]")

                table.add_row(str(idx), name, param, quant, size_gb, ", ".join(roles) if roles else "[dim]available[/dim]")
            console.print(table)
        else:
            console.print("[yellow]⚠ Ollama is offline or has no models installed.[/yellow]")

        console.print(
            "\n[bold]Model Actions:[/bold]\n"
            "  [1] Set Default Agent Model (General Agents & Literature Synthesis)\n"
            "  [2] Set Reasoning Model (Hypothesis Formulation & Adversarial Tribunal)\n"
            "  [3] Set Embedding Model (Vector Semantic Indexing)\n"
            "  [4] Change Ollama Endpoint URL\n"
            "  [5] Test Active Model with Live Inference Prompt\n"
            "  [6] Pull / Download New Model from Ollama\n"
            "  [7] Save Current Model Configuration Permanently to .env\n"
            "  [0] Return to Main Menu\n"
        )

        choice = Prompt.ask("[bold]Select an action[/bold]", default="0")
        if choice == "0":
            break

        model_names = [m.get("name", "") for m in models]

        if choice in ["1", "2", "3"]:
            role_labels = {
                "1": ("Default Agent Model", "OLLAMA_DEFAULT_MODEL", settings.ollama_default_model),
                "2": ("Reasoning Model", "OLLAMA_REASONING_MODEL", settings.ollama_reasoning_model),
                "3": ("Embedding Model", "OLLAMA_EMBEDDING_MODEL", settings.ollama_embedding_model),
            }
            label, key, cur_val = role_labels[choice]
            console.print(f"\nCurrent {label}: [bold cyan]{cur_val}[/bold cyan]")
            if models:
                console.print("Choose from installed models by index number, or enter a custom tag:")
                for i, m in enumerate(models, 1):
                    console.print(f"  [{i}] {m.get('name')}")
                val_input = Prompt.ask("Enter model number or exact model name", default=cur_val)
                if val_input.isdigit() and 1 <= int(val_input) <= len(models):
                    selected = models[int(val_input) - 1].get("name", "")
                else:
                    selected = val_input.strip()
            else:
                selected = Prompt.ask("Enter model tag (e.g. llama3.2:1b)", default=cur_val).strip()

            if selected:
                update_settings({key: selected}, persist=True)
                console.print(f"[bold green]✓ Updated {label} to:[/bold green] [cyan]{selected}[/cyan] (Saved to .env)")

        elif choice == "4":
            new_url = Prompt.ask("Enter Ollama Endpoint URL", default=settings.ollama_base_url).strip()
            if new_url:
                update_settings({"OLLAMA_BASE_URL": new_url}, persist=True)
                client.base_url = new_url
                console.print(f"[bold green]✓ Ollama URL updated to {new_url} (Saved to .env)[/bold green]")

        elif choice == "5":
            prompt_text = Prompt.ask(
                "Enter test prompt",
                default="Explain the mechanism of action of trastuzumab in HER2-positive cancer in 2 sentences.",
            )
            cmd_test_chat(client, prompt_text)

        elif choice == "6":
            pull_tag = Prompt.ask("Enter model tag to pull (e.g. llama3.2:1b, llama3.2:3b, qwen2.5:7b)")
            if pull_tag:
                console.print(f"[cyan]Pulling model [bold]{pull_tag}[/bold] via Ollama...[/cyan]")
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
                    task = prog.add_task(f"Downloading {pull_tag}...", total=None)
                    try:
                        success = client.pull_model(pull_tag)
                        prog.remove_task(task)
                        if success:
                            console.print(f"[bold green]✓ Successfully installed {pull_tag}![/bold green]")
                        else:
                            console.print(f"[bold red]✗ Failed to pull {pull_tag}. Check server logs.[/bold red]")
                    except Exception as ex:
                        prog.remove_task(task)
                        console.print(f"[bold red]Error pulling model:[/bold red] {ex}")

        elif choice == "7":
            save_env_variables(
                {
                    "OLLAMA_DEFAULT_MODEL": settings.ollama_default_model,
                    "OLLAMA_REASONING_MODEL": settings.ollama_reasoning_model,
                    "OLLAMA_EMBEDDING_MODEL": settings.ollama_embedding_model,
                    "OLLAMA_BASE_URL": settings.ollama_base_url,
                }
            )
            console.print("[bold green]✓ Model configuration successfully written to .env file![/bold green]")


def menu_configure_parameters() -> None:
    """Interactive submenu for configuring research parameters and module toggles."""
    while True:
        settings = get_settings()
        console.rule("[bold cyan]Configure Research Parameters & In-Silico Modules[/bold cyan]")

        table = Table(title="Platform Research & In-Silico Pipeline Settings", header_style="bold magenta")
        table.add_column("#", justify="center", style="bold")
        table.add_column("Parameter / Feature", style="cyan")
        table.add_column("Current Setting", style="bold white")
        table.add_column("Description", style="dim")

        table.add_row("1", "Investigation Depth", settings.default_depth.upper(), "deep (2 cycles), exhaustive (3 cycles), institutional (4 cycles)")
        table.add_row("2", "Literature Limit (Papers)", str(settings.default_literature_limit), "Target count of peer-reviewed papers to ingest per topic")
        table.add_row("3", "Clinical Trials Limit", str(settings.default_trials_limit), "Human clinical trials to fetch from ClinicalTrials.gov")
        table.add_row("4", "Max Hypotheses per Cycle", str(settings.default_max_hypotheses), "Maximum hypotheses formulated by Dr. Hypothesis")
        table.add_row("5", "DepMap Cancer Dependency", "[green]ENABLED[/green]" if settings.enable_depmap else "[red]DISABLED[/red]", "Functional genomics Chronos score verification across cell lines")
        table.add_row("6", "openFDA FAERS Surveillance", "[green]ENABLED[/green]" if settings.enable_pharmacovigilance else "[red]DISABLED[/red]", "Real-world adverse event monitoring from human patient reports")
        table.add_row("7", "NCBI ClinVar Stratification", "[green]ENABLED[/green]" if settings.enable_clinvar else "[red]DISABLED[/red]", "Pathogenic variant analysis & genomic inclusion/exclusion criteria")
        table.add_row("8", "Single-Cell TME Interactome", "[green]ENABLED[/green]" if settings.enable_tme else "[red]DISABLED[/red]", "Tumor microenvironment cell deconvolution & paracrine axes")
        table.add_row("9", "CRISPR sgRNA & Primer Design", "[green]ENABLED[/green]" if settings.enable_crispr else "[red]DISABLED[/red]", "Laboratory knockout guide RNA and PCR validation primer synthesis")
        table.add_row("10", "Peer-Review Tribunal", "[green]ENABLED[/green]" if settings.enable_tribunal else "[red]DISABLED[/red]", "Adversarial 3-member expert evaluation and scoring board")
        table.add_row("11", "Export Formats", settings.export_formats.upper(), "Comma-separated export formats (Markdown, PDF, LaTeX, BibTeX)")
        table.add_row("12", "NCBI Email & API Key", f"{settings.ncbi_email} / {'Configured' if settings.ncbi_api_key else 'None'}", "E-utilities query credentials")
        table.add_row("13", "Resource Guard (Anti-Freeze)", "[green]ENABLED[/green]" if settings.enable_resource_guard else "[red]DISABLED[/red]", "Host CPU thread reservation & anti-freeze memory throttling")
        table.add_row("14", "Max Safe Papers Cap", str(settings.max_safe_papers_limit), "Maximum literature papers per cycle to prevent memory exhaustion")
        table.add_row("15", "Max Safe Trials Cap", str(settings.max_safe_trials_limit), "Maximum clinical trials per cycle to protect API limits")

        console.print(table)
        console.print(
            "\n[bold]Options:[/bold]\n"
            "  Enter a number [bold][1-15][/bold] to toggle or edit that parameter.\n"
            "  [bold][S][/bold] Save all current settings to .env file permanently\n"
            "  [bold][0][/bold] Return to Main Menu\n"
        )

        choice = Prompt.ask("[bold]Select option[/bold]", default="0").strip().upper()
        if choice == "0":
            break

        if choice == "1":
            new_depth = Prompt.ask("Choose depth level", choices=["deep", "exhaustive", "institutional"], default=settings.default_depth)
            update_settings({"DEFAULT_DEPTH": new_depth}, persist=True)
            console.print(f"[bold green]✓ Updated depth to:[/bold green] [cyan]{new_depth}[/cyan]")

        elif choice == "2":
            new_limit = IntPrompt.ask("Enter max papers to analyze", default=settings.default_literature_limit)
            update_settings({"DEFAULT_LITERATURE_LIMIT": str(new_limit)}, persist=True)
            console.print(f"[bold green]✓ Updated papers limit to:[/bold green] [cyan]{new_limit}[/cyan]")

        elif choice == "3":
            new_trials = IntPrompt.ask("Enter max clinical trials to analyze", default=settings.default_trials_limit)
            update_settings({"DEFAULT_TRIALS_LIMIT": str(new_trials)}, persist=True)
            console.print(f"[bold green]✓ Updated clinical trials limit to:[/bold green] [cyan]{new_trials}[/cyan]")

        elif choice == "4":
            new_hyp = IntPrompt.ask("Enter max hypotheses formulated per cycle", default=settings.default_max_hypotheses)
            update_settings({"DEFAULT_MAX_HYPOTHESES": str(new_hyp)}, persist=True)
            console.print(f"[bold green]✓ Updated hypotheses limit to:[/bold green] [cyan]{new_hyp}[/cyan]")

        elif choice == "5":
            new_val = not settings.enable_depmap
            update_settings({"ENABLE_DEPMAP": str(new_val)}, persist=True)
            console.print(f"[bold green]✓ DepMap verification is now {'ENABLED' if new_val else 'DISABLED'}[/bold green]")

        elif choice == "6":
            new_val = not settings.enable_pharmacovigilance
            update_settings({"ENABLE_PHARMACOVIGILANCE": str(new_val)}, persist=True)
            console.print(f"[bold green]✓ openFDA FAERS surveillance is now {'ENABLED' if new_val else 'DISABLED'}[/bold green]")

        elif choice == "7":
            new_val = not settings.enable_clinvar
            update_settings({"ENABLE_CLINVAR": str(new_val)}, persist=True)
            console.print(f"[bold green]✓ ClinVar stratification is now {'ENABLED' if new_val else 'DISABLED'}[/bold green]")

        elif choice == "8":
            new_val = not settings.enable_tme
            update_settings({"ENABLE_TME": str(new_val)}, persist=True)
            console.print(f"[bold green]✓ Single-Cell TME Interactome is now {'ENABLED' if new_val else 'DISABLED'}[/bold green]")

        elif choice == "9":
            new_val = not settings.enable_crispr
            update_settings({"ENABLE_CRISPR": str(new_val)}, persist=True)
            console.print(f"[bold green]✓ CRISPR sgRNA & primer design is now {'ENABLED' if new_val else 'DISABLED'}[/bold green]")

        elif choice == "10":
            new_val = not settings.enable_tribunal
            update_settings({"ENABLE_TRIBUNAL": str(new_val)}, persist=True)
            console.print(f"[bold green]✓ Adversarial Peer-Review Tribunal is now {'ENABLED' if new_val else 'DISABLED'}[/bold green]")

        elif choice == "11":
            new_exports = Prompt.ask("Enter comma-separated export formats", default=settings.export_formats)
            update_settings({"EXPORT_FORMATS": new_exports}, persist=True)
            console.print(f"[bold green]✓ Export formats updated to:[/bold green] [cyan]{new_exports}[/cyan]")

        elif choice == "12":
            new_email = Prompt.ask("Enter NCBI Contact Email", default=settings.ncbi_email)
            new_key = Prompt.ask("Enter NCBI API Key (or blank)", default=settings.ncbi_api_key or "")
            update_settings({"NCBI_EMAIL": new_email, "NCBI_API_KEY": new_key}, persist=True)
            console.print("[bold green]✓ NCBI credentials saved successfully.[/bold green]")

        elif choice == "13":
            new_val = not settings.enable_resource_guard
            update_settings({"ENABLE_RESOURCE_GUARD": str(new_val)}, persist=True)
            console.print(f"[bold green]✓ Resource Guard is now {'ENABLED' if new_val else 'DISABLED'}[/bold green]")

        elif choice == "14":
            new_cap = IntPrompt.ask("Enter max safe papers cap per cycle", default=settings.max_safe_papers_limit)
            update_settings({"MAX_SAFE_PAPERS_LIMIT": str(new_cap)}, persist=True)
            console.print(f"[bold green]✓ Max safe papers limit updated to:[/bold green] [cyan]{new_cap}[/cyan]")

        elif choice == "15":
            new_cap = IntPrompt.ask("Enter max safe trials cap per cycle", default=settings.max_safe_trials_limit)
            update_settings({"MAX_SAFE_TRIALS_LIMIT": str(new_cap)}, persist=True)
            console.print(f"[bold green]✓ Max safe trials limit updated to:[/bold green] [cyan]{new_cap}[/cyan]")

        elif choice == "S":
            save_env_variables(
                {
                    "DEFAULT_DEPTH": settings.default_depth,
                    "DEFAULT_LITERATURE_LIMIT": str(settings.default_literature_limit),
                    "DEFAULT_TRIALS_LIMIT": str(settings.default_trials_limit),
                    "DEFAULT_MAX_HYPOTHESES": str(settings.default_max_hypotheses),
                    "ENABLE_DEPMAP": str(settings.enable_depmap),
                    "ENABLE_PHARMACOVIGILANCE": str(settings.enable_pharmacovigilance),
                    "ENABLE_CLINVAR": str(settings.enable_clinvar),
                    "ENABLE_TME": str(settings.enable_tme),
                    "ENABLE_CRISPR": str(settings.enable_crispr),
                    "ENABLE_TRIBUNAL": str(settings.enable_tribunal),
                    "EXPORT_FORMATS": settings.export_formats,
                    "ENABLE_RESOURCE_GUARD": str(settings.enable_resource_guard),
                    "MAX_SAFE_PAPERS_LIMIT": str(settings.max_safe_papers_limit),
                    "MAX_SAFE_TRIALS_LIMIT": str(settings.max_safe_trials_limit),
                }
            )
            console.print("[bold green]✓ All settings saved to .env file.[/bold green]")


def menu_run_investigation(client: OllamaClient) -> None:
    """Guided workflow for launching an autonomous scientific investigation."""
    settings = get_settings()
    console.rule("[bold green]Launch Autonomous Scientific Investigation[/bold green]")

    examples = [
        "Overcoming CD44v6-driven dormancy in triple-negative breast cancer",
        "Targeting glioblastoma stem cells via STAT3 and blood-brain barrier bypass",
        "Synergistic KRAS G12D inhibition and PD-L1 immune checkpoint blockade",
    ]

    console.print("[bold]Enter a disease, molecular target, or biomedical research question:[/bold]")
    for i, ex in enumerate(examples, 1):
        console.print(f"  [dim][{i}] Example: {ex}[/dim]")

    topic = Prompt.ask("\n[bold cyan]Research Topic[/bold cyan]", default=examples[0]).strip()
    if topic in ["1", "2", "3"]:
        topic = examples[int(topic) - 1]

    console.print(
        f"\n[bold]Proposed Investigation Plan:[/bold]\n"
        f"  • Topic: [bold yellow]{topic}[/bold yellow]\n"
        f"  • Model: [cyan]{settings.ollama_default_model}[/cyan]\n"
        f"  • Depth: [magenta]{settings.default_depth.upper()}[/magenta]\n"
        f"  • Ingest Papers: [cyan]{settings.default_literature_limit}[/cyan] | Clinical Trials: [cyan]{settings.default_trials_limit}[/cyan]\n"
        f"  • Modules: DepMap ({'ON' if settings.enable_depmap else 'OFF'}), "
        f"openFDA ({'ON' if settings.enable_pharmacovigilance else 'OFF'}), "
        f"ClinVar ({'ON' if settings.enable_clinvar else 'OFF'}), "
        f"TME ({'ON' if settings.enable_tme else 'OFF'}), "
        f"CRISPR ({'ON' if settings.enable_crispr else 'OFF'}), "
        f"Tribunal ({'ON' if settings.enable_tribunal else 'OFF'})\n"
        f"  • Exports: [blue]{settings.export_formats}[/blue]\n"
    )

    console.print(
        "  [1] Start Investigation in Foreground (Live Terminal)\n"
        "  [2] Start in Background (PM2-style daemon, safe to disconnect SSH)\n"
        "  [3] Customize Parameters Before Launch\n"
        "  [0] Cancel & Return\n"
    )

    action = Prompt.ask("Choose option", default="1")
    if action == "0":
        return

    papers = settings.default_literature_limit
    trials = settings.default_trials_limit
    depth = settings.default_depth
    model = settings.ollama_default_model
    run_in_background = (action == "2")

    if action == "3":
        depth = Prompt.ask("Select Depth", choices=["deep", "exhaustive", "institutional"], default=settings.default_depth)
        papers = IntPrompt.ask("Papers count", default=settings.default_literature_limit)
        trials = IntPrompt.ask("Trials count", default=settings.default_trials_limit)

        installed_models = fetch_installed_models(client)
        if installed_models:
            avail_names = [m["name"] for m in installed_models if "embed" not in m["name"].lower()]
            console.print(f"[dim]Verified installed models in Ollama: {', '.join(avail_names)}[/dim]")

        model_input = Prompt.ask("Model Override (or blank for default)", default=settings.ollama_default_model).strip()
        from helixmind.core.resources import get_resource_manager
        guard = get_resource_manager()
        model, warning = guard.resolve_safe_model(client, model_input)
        if warning:
            console.print(Panel(warning, border_style="yellow"))

        run_in_background = Confirm.ask("Run detached in background (PM2-style daemon)?", default=True)

    if run_in_background:
        cmd_start_daemon(
            topic=topic,
            papers=papers,
            trials=trials,
            depth=depth,
            model=model,
        )
    else:
        cmd_investigate(
            topic=topic,
            max_papers=papers,
            max_trials=trials,
            depth=depth,
            model=model,
        )

    Prompt.ask("\n[bold dim]Press Enter to return to main menu[/bold dim]", default="")


def menu_copilot_chat(client: OllamaClient) -> None:
    """Interactive terminal chat session with Dr. Copilot."""
    settings = get_settings()
    console.rule("[bold magenta]Interactive Research Copilot (Dr. Copilot)[/bold magenta]")
    console.print(
        "Chat in real time with HelixMind's Co-Investigator intelligence.\n"
        "Ask mechanistic questions, cite scientific literature, steer hypotheses, or design CRISPR protocols.\n"
        "[dim]Type '/exit' or 'exit' to return to the main menu. Type '/clear' to reset chat memory.[/dim]\n"
    )

    copilot = ResearchCopilot(ollama_client=client, model=settings.ollama_default_model)
    session_id = f"CLI-CHAT-{int(time.time())}"

    while True:
        try:
            user_msg = Prompt.ask("\n[bold cyan]Dr. Copilot > User[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_msg:
            continue
        if user_msg.lower() in ["/exit", "exit", "quit", "/quit", "/back"]:
            break
        if user_msg.lower() == "/clear":
            copilot._chat_histories[session_id] = []
            console.print("[dim]Chat history cleared.[/dim]")
            continue

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
            task = prog.add_task("Dr. Copilot is reasoning and assembling evidence...", total=None)
            try:
                resp = asyncio.run(copilot.chat_async(investigation_id=session_id, user_message=user_msg))
            except Exception as ex:
                prog.remove_task(task)
                console.print(f"[bold red]Copilot error:[/bold red] {ex}")
                continue
            prog.remove_task(task)

        console.print(f"\n[bold green]Dr. Copilot:[/bold green]\n{resp.reply}\n")
        if resp.citations:
            console.print(f"[dim]📚 Citations Referenced: {', '.join(resp.citations)}[/dim]")
        if resp.actions_taken:
            console.print(f"[dim]⚡ Triggered In-Silico Actions: {', '.join(resp.actions_taken)}[/dim]")


def menu_insilico_tools() -> None:
    """Interactive suite for standalone molecular and genomic computations."""
    while True:
        console.rule("[bold cyan]Standalone In-Silico Genomic & Molecular Suite[/bold cyan]")
        console.print(
            "Execute instantaneous biomedical queries without running a full research cycle:\n\n"
            "  [1] 🧬 DepMap Functional Genomics (Chronos score & cancer cell dependency)\n"
            "  [2] 💊 openFDA Real-World Adverse Reactions (FAERS post-marketing surveillance)\n"
            "  [3] 🧬 NCBI ClinVar Precision Patient Stratification (Pathogenic mutations & criteria)\n"
            "  [4] 🔬 Single-Cell Tumor Microenvironment (TME interactome & paracrine axes)\n"
            "  [5] ✂️  CRISPR-Cas9 sgRNA & Golden Gate PCR Primer Design\n"
            "  [6] 🧪 De Novo Molecular Analog Generator & MMFF94 Binding Affinity\n"
            "  [7] 📊 Virtual Patient Phase 2b Trial Simulator (Kaplan-Meier survival curves)\n"
            "  [0] Return to Main Menu\n"
        )

        choice = Prompt.ask("[bold]Select in-silico tool[/bold]", default="0")
        if choice == "0":
            break

        if choice == "1":
            gene = Prompt.ask("Enter Gene Symbol (e.g. EGFR, STAT3, CD44, TP53)", default="EGFR")
            cond = Prompt.ask("Enter Target Disease (e.g. Glioblastoma, Breast Cancer)", default="Glioblastoma")
            run_tool_depmap(gene, cond)

        elif choice == "2":
            drug = Prompt.ask("Enter Drug or Monoclonal Antibody Name", default="Trastuzumab")
            run_tool_openfda(drug)

        elif choice == "3":
            gene = Prompt.ask("Enter Gene Symbol", default="EGFR")
            cond = Prompt.ask("Enter Condition", default="Glioblastoma")
            run_tool_clinvar(gene, cond)

        elif choice == "4":
            genes_str = Prompt.ask("Enter Target Genes (comma-separated)", default="CD274, STAT3, TGFB1")
            cond = Prompt.ask("Enter Target Condition", default="Glioblastoma")
            run_tool_tme(genes_str, cond)

        elif choice == "5":
            gene = Prompt.ask("Enter Target Gene Symbol", default="CD274")
            run_tool_crispr(gene)

        elif choice == "6":
            drug = Prompt.ask("Enter Parent Drug Name or Class (e.g. Tamoxifen, Imatinib, Gefitinib)", default="Tamoxifen")
            run_tool_denovo(drug)

        elif choice == "7":
            drug = Prompt.ask("Enter Investigational Drug Name", default="Trastuzumab")
            cond = Prompt.ask("Enter Oncology Condition", default="Breast Cancer")
            run_tool_clinical_simulation(drug, cond)

        Prompt.ask("\n[bold dim]Press Enter to continue[/bold dim]", default="")


def menu_system_diagnostics(client: OllamaClient) -> None:
    """Comprehensive connectivity and health diagnostics for all platform endpoints."""
    console.rule("[bold cyan]System Diagnostics & Biomedical API Health[/bold cyan]")
    settings = get_settings()

    endpoints = [
        ("Ollama Local Server", client.base_url + "/api/tags", "GET"),
        ("NCBI E-utilities (PubMed)", f"{settings.ncbi_base_url}/einfo.fcgi?retmode=json", "GET"),
        ("ClinicalTrials.gov (API v2)", f"{settings.clinical_trials_base_url}/studies?pageSize=1", "GET"),
        ("Europe PMC REST API", f"{settings.europe_pmc_base_url}/search?query=cancer&pageSize=1&format=json", "GET"),
        ("openFDA FAERS Drug Events", "https://api.fda.gov/drug/event.json?limit=1", "GET"),
    ]

    table = Table(title="Live Biomedical Gateway Connectivity & Latency", header_style="bold magenta")
    table.add_column("Service Name", style="bold cyan")
    table.add_column("Status", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Endpoint", style="dim")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as prog:
        task = prog.add_task("Pinging biomedical databases...", total=len(endpoints))
        for name, url, method in endpoints:
            t0 = time.time()
            try:
                with httpx.Client(timeout=6.0) as http:
                    res = http.request(method, url)
                    elapsed_ms = (time.time() - t0) * 1000.0
                    if res.status_code in [200, 400]:  # openFDA returns 400 if search syntax empty, meaning endpoint is up
                        table.add_row(name, "[bold green]ONLINE[/bold green]", f"{elapsed_ms:.0f} ms", url[:45] + "...")
                    else:
                        table.add_row(name, f"[yellow]HTTP {res.status_code}[/yellow]", f"{elapsed_ms:.0f} ms", url[:45] + "...")
            except Exception as ex:
                table.add_row(name, "[bold red]OFFLINE[/bold red]", "Timeout", str(ex)[:35])
            prog.update(task, advance=1)

    console.print(table)

    # Inspect storage
    papers_count = len(list((settings.data_dir / "papers").glob("*.md"))) if (settings.data_dir / "papers").exists() else 0
    reports_count = len(list((settings.data_dir / "reports").glob("*.json"))) if (settings.data_dir / "reports").exists() else 0

    console.print(
        Panel.fit(
            f"[bold]Cache Directory:[/bold] {settings.cache_dir}\n"
            f"[bold]Data Storage Directory:[/bold] {settings.data_dir}\n"
            f"[bold]Publication Papers Authored:[/bold] [green]{papers_count}[/green]\n"
            f"[bold]Investigation Reports Generated:[/bold] [cyan]{reports_count}[/cyan]",
            title="Local Storage Metrics",
            border_style="blue",
        )
    )

    Prompt.ask("\n[bold dim]Press Enter to return to main menu[/bold dim]", default="")


def menu_start_server() -> None:
    """Launch the FastAPI and WebSockets server interactively."""
    console.rule("[bold cyan]Launch HelixMind API Gateway Server[/bold cyan]")
    host = Prompt.ask("Bind Host", default="0.0.0.0")
    port = IntPrompt.ask("Bind Port", default=8000)
    reload = Confirm.ask("Enable Hot Reloading (development mode)?", default=False)

    console.print(f"[bold green]Starting HelixMind API Server on http://{host}:{port}...[/bold green]")
    start_server(host=host, port=port, reload=reload)


# =============================================================================
# Interactive Master Menu Loop
# =============================================================================


def interactive_main_menu(client: OllamaClient) -> None:
    """Interactive CLI menu driver."""
    while True:
        display_banner()
        console.print(get_system_status_panel(client))

        console.print(
            "\n[bold magenta]HelixMind Main Menu:[/bold magenta]\n"
            "  [bold cyan][1][/bold cyan] 🚀 Launch Autonomous Scientific Investigation\n"
            "  [bold cyan][2][/bold cyan] 🤖 Configure AI Models & LLM Providers\n"
            "  [bold cyan][3][/bold cyan] ⚙️  Configure Parameters & Pipeline Modules\n"
            "  [bold cyan][4][/bold cyan] 💬 Interactive Research Copilot (Dr. Copilot)\n"
            "  [bold cyan][5][/bold cyan] 🔬 Standalone In-Silico Molecular & Genomic Suite\n"
            "  [bold cyan][6][/bold cyan] 🩺 System Diagnostics & Biomedical API Health\n"
            "  [bold cyan][7][/bold cyan] 🌐 Start REST & WebSocket API Gateway Server\n"
            "  [bold cyan][8][/bold cyan] ⚡ Background Missions & Process Manager (PM2-style)\n"
            "  [bold red][0][/bold red] 🚪 Exit\n"
        )

        try:
            choice = Prompt.ask("[bold]Select an option[/bold]", default="1").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting HelixMind CLI. Goodbye![/dim]")
            sys.exit(0)

        if choice == "0":
            console.print("\n[dim]Exiting HelixMind CLI. Goodbye![/dim]")
            sys.exit(0)
        elif choice == "1":
            menu_run_investigation(client)
        elif choice == "2":
            menu_configure_models(client)
        elif choice == "3":
            menu_configure_parameters()
        elif choice == "4":
            menu_copilot_chat(client)
        elif choice == "5":
            menu_insilico_tools()
        elif choice == "6":
            menu_system_diagnostics(client)
        elif choice == "7":
            menu_start_server()
        elif choice == "8":
            cmd_ps()
            console.print("\n[bold]Options:[/bold] [1] Stream logs  [2] Stop a mission  [0] Return")
            sub = Prompt.ask("Choose", default="0")
            if sub == "1":
                inv = Prompt.ask("Enter Investigation ID or PID")
                cmd_logs(inv, follow=True)
            elif sub == "2":
                inv = Prompt.ask("Enter Investigation ID or PID to stop")
                cmd_stop(inv)
        else:
            console.print("[yellow]Invalid choice, please select an option from 0 to 8.[/yellow]")


# =============================================================================
# CLI Entrypoint & Argument Parser
# =============================================================================


def main() -> None:
    """CLI entrypoint supporting both headless subcommands and interactive menu."""
    parser = argparse.ArgumentParser(
        description="HelixMind Autonomous Scientific Research Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive terminal menu",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: menu (explicit interactive menu)
    subparsers.add_parser("menu", help="Launch interactive terminal menu")

    # Command: status
    subparsers.add_parser("status", help="Check Ollama connection and list installed models")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Test chat completion with Ollama")
    chat_parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="Explain the role of tau phosphorylation in Alzheimer's disease in 3 sentences.",
        help="Prompt to send to model",
    )
    chat_parser.add_argument("-m", "--model", type=str, default=None, help="Model override")

    # Command: embed
    embed_parser = subparsers.add_parser("embed", help="Test vector embeddings with Ollama")
    embed_parser.add_argument(
        "-t", "--text",
        type=str,
        default="Amyloid-beta plaques and neurofibrillary tangles in neurodegeneration.",
        help="Text to embed",
    )
    embed_parser.add_argument("-m", "--model", type=str, default=None, help="Embedding model override")

    # Command: investigate (Auto-Pilot multi-agent research loop)
    inv_parser = subparsers.add_parser("investigate", help="Run autonomous multi-agent scientific investigation")
    inv_parser.add_argument(
        "topic",
        type=str,
        help="Disease, cellular target, or biological topic to investigate (e.g. 'breast cancer immunotherapy')",
    )
    inv_parser.add_argument(
        "--papers",
        type=int,
        default=None,
        help="Max papers to fetch and index (default: configured default)",
    )
    inv_parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Max clinical trials to fetch",
    )
    inv_parser.add_argument(
        "--depth",
        type=str,
        choices=["deep", "exhaustive", "institutional"],
        default=None,
        help="Depth of autonomous investigation: deep (2 cycles, default), exhaustive (3 cycles), institutional (4 cycles)",
    )
    inv_parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="LLM model override (e.g. 'llama3.2:1b' or 'llama3.2:3b')",
    )
    inv_parser.add_argument("--no-depmap", action="store_true", help="Disable DepMap cancer dependency verification")
    inv_parser.add_argument("--no-fda", action="store_true", help="Disable openFDA FAERS adverse event monitoring")
    inv_parser.add_argument("--no-clinvar", action="store_true", help="Disable NCBI ClinVar patient stratification")
    inv_parser.add_argument("--no-tme", action="store_true", help="Disable single-cell TME interactome mapping")
    inv_parser.add_argument("--no-crispr", action="store_true", help="Disable CRISPR sgRNA & primer design")
    inv_parser.add_argument("--no-tribunal", action="store_true", help="Disable adversarial peer-review tribunal")
    inv_parser.add_argument(
        "-d", "--daemon", "--background", "-b",
        dest="background",
        action="store_true",
        help="Run detached in background like PM2 (safe to disconnect SSH session)",
    )
    inv_parser.add_argument(
        "--export-formats",
        type=str,
        default=None,
        help="Comma-separated list of formats (markdown, pdf, latex, bibtex)",
    )

    # Command: start (Background daemon like PM2)
    start_parser = subparsers.add_parser(
        "start",
        help="Launch autonomous investigation detached in background (like PM2, safe to disconnect SSH)",
    )
    start_parser.add_argument(
        "topic",
        type=str,
        help="Disease, cellular target, or biological topic to investigate",
    )
    start_parser.add_argument("--papers", type=int, default=None, help="Max papers to fetch and index")
    start_parser.add_argument("--trials", type=int, default=None, help="Max clinical trials to fetch")
    start_parser.add_argument("--depth", type=str, choices=["deep", "exhaustive", "institutional"], default="deep")
    start_parser.add_argument("-m", "--model", type=str, default=None, help="LLM model override")
    start_parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    start_parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    start_parser.add_argument("--no-depmap", action="store_true", help="Disable DepMap")
    start_parser.add_argument("--no-fda", action="store_true", help="Disable openFDA")
    start_parser.add_argument("--no-clinvar", action="store_true", help="Disable ClinVar")
    start_parser.add_argument("--no-tme", action="store_true", help="Disable TME")
    start_parser.add_argument("--no-crispr", action="store_true", help="Disable CRISPR")
    start_parser.add_argument("--no-tribunal", action="store_true", help="Disable tribunal")

    # Command: ps / list (PM2-style process list)
    subparsers.add_parser(
        "ps",
        aliases=["list"],
        help="List all background investigations and server status (like 'pm2 ps')",
    )

    # Command: logs (PM2-style log stream)
    logs_parser = subparsers.add_parser(
        "logs",
        help="View or stream logs from a background investigation (like 'pm2 logs')",
    )
    logs_parser.add_argument("id", type=str, help="Investigation ID or Process PID")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow / stream log output in real-time")
    logs_parser.add_argument("-n", "--lines", type=int, default=100, help="Number of trailing lines to show")

    # Command: stop (PM2-style stop)
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop a background investigation process (like 'pm2 stop')",
    )
    stop_parser.add_argument("id", type=str, help="Investigation ID or Process PID to terminate")

    # Command: copilot (Interactive terminal co-investigator)
    subparsers.add_parser("copilot", help="Launch interactive Dr. Copilot terminal chat session")

    # Command: tools (Standalone in-silico tools)
    tools_parser = subparsers.add_parser("tools", help="Run standalone in-silico genomic & molecular tools")
    tool_subparsers = tools_parser.add_subparsers(dest="tool_name", help="Tool to execute")

    tool_depmap = tool_subparsers.add_parser("depmap", help="Query DepMap cancer essentiality")
    tool_depmap.add_argument("gene", type=str, help="Gene symbol (e.g. EGFR)")
    tool_depmap.add_argument("--condition", type=str, default="Oncology", help="Cancer condition")

    tool_fda = tool_subparsers.add_parser("fda", help="Query openFDA FAERS adverse reactions")
    tool_fda.add_argument("drug", type=str, help="Drug name (e.g. Trastuzumab)")

    tool_clinvar = tool_subparsers.add_parser("clinvar", help="Query NCBI ClinVar pathogenic variants")
    tool_clinvar.add_argument("gene", type=str, help="Gene symbol (e.g. EGFR)")
    tool_clinvar.add_argument("--condition", type=str, default="Oncology", help="Target condition")

    tool_tme = tool_subparsers.add_parser("tme", help="Deconvolve single-cell tumor microenvironment")
    tool_tme.add_argument("genes", type=str, help="Comma-separated gene symbols (e.g. STAT3,CD274)")
    tool_tme.add_argument("--condition", type=str, default="Glioblastoma", help="Target disease")

    tool_crispr = tool_subparsers.add_parser("crispr", help="Design CRISPR-Cas9 sgRNAs and validation primers")
    tool_crispr.add_argument("gene", type=str, help="Target gene symbol (e.g. CD274)")

    tool_denovo = tool_subparsers.add_parser("denovo", help="Generate optimized bioisosteric chemical analogs")
    tool_denovo.add_argument("parent", type=str, help="Parent drug name (e.g. Tamoxifen)")

    tool_sim = tool_subparsers.add_parser("trial", help="Simulate virtual clinical trial with Kaplan-Meier survival")
    tool_sim.add_argument("drug", type=str, help="Investigational drug name")
    tool_sim.add_argument("--condition", type=str, default="Breast Cancer", help="Target condition")

    # Command: config (View or update configuration)
    config_parser = subparsers.add_parser("config", help="View or update configuration settings")
    config_parser.add_argument("--list", action="store_true", help="List all current configuration values")
    config_parser.add_argument("--set-default-model", type=str, default=None, help="Update default LLM model")
    config_parser.add_argument("--set-reasoning-model", type=str, default=None, help="Update reasoning LLM model")
    config_parser.add_argument("--set-embedding-model", type=str, default=None, help="Update embedding model")
    config_parser.add_argument("--set-depth", type=str, choices=["deep", "exhaustive", "institutional"], default=None, help="Set default depth")
    config_parser.add_argument("--set-url", type=str, default=None, help="Set Ollama base URL")

    # Command: serve (Launch FastAPI & WebSockets gateway)
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST & real-time WebSockets API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable hot reloading")

    args = parser.parse_args()
    client = OllamaClient()

    # Route: Interactive menu if requested or if no arguments provided in interactive terminal
    if args.interactive or args.command == "menu":
        interactive_main_menu(client)
        return

    if args.command is None:
        if sys.stdin.isatty():
            interactive_main_menu(client)
        else:
            parser.print_help()
        return

    # Route: Subcommands
    if args.command == "status":
        cmd_status(client)
    elif args.command == "chat":
        cmd_test_chat(client, prompt=args.prompt, model=args.model)
    elif args.command == "embed":
        cmd_test_embed(client, sample_text=args.text, model=args.model)
    elif args.command == "copilot":
        menu_copilot_chat(client)
    elif args.command == "start":
        cmd_start_daemon(
            topic=args.topic,
            papers=args.papers,
            trials=args.trials,
            depth=args.depth,
            model=args.model,
            host=args.host,
            port=args.port,
            enable_depmap=False if args.no_depmap else None,
            enable_pharmacovigilance=False if args.no_fda else None,
            enable_clinvar=False if args.no_clinvar else None,
            enable_tme=False if args.no_tme else None,
            enable_crispr=False if args.no_crispr else None,
            enable_tribunal=False if args.no_tribunal else None,
        )
    elif args.command in ("ps", "list"):
        cmd_ps()
    elif args.command == "logs":
        cmd_logs(args.id, follow=args.follow, lines=args.lines)
    elif args.command == "stop":
        cmd_stop(args.id)
    elif args.command == "investigate":
        if getattr(args, "background", False):
            cmd_start_daemon(
                topic=args.topic,
                papers=args.papers,
                trials=args.trials,
                depth=args.depth,
                model=args.model,
                enable_depmap=False if args.no_depmap else None,
                enable_pharmacovigilance=False if args.no_fda else None,
                enable_clinvar=False if args.no_clinvar else None,
                enable_tme=False if args.no_tme else None,
                enable_crispr=False if args.no_crispr else None,
                enable_tribunal=False if args.no_tribunal else None,
            )
        else:
            export_fmts = [f.strip() for f in args.export_formats.split(",") if f.strip()] if args.export_formats else None
            cmd_investigate(
                topic=args.topic,
                max_papers=args.papers,
                max_trials=args.trials,
                depth=args.depth,
                model=args.model,
                enable_depmap=False if args.no_depmap else None,
                enable_pharmacovigilance=False if args.no_fda else None,
                enable_clinvar=False if args.no_clinvar else None,
                enable_tme=False if args.no_tme else None,
                enable_crispr=False if args.no_crispr else None,
                enable_tribunal=False if args.no_tribunal else None,
                export_formats=export_fmts,
            )
    elif args.command == "tools":
        if args.tool_name == "depmap":
            run_tool_depmap(args.gene, args.condition)
        elif args.tool_name == "fda":
            run_tool_openfda(args.drug)
        elif args.tool_name == "clinvar":
            run_tool_clinvar(args.gene, args.condition)
        elif args.tool_name == "tme":
            run_tool_tme(args.genes, args.condition)
        elif args.tool_name == "crispr":
            run_tool_crispr(args.gene)
        elif args.tool_name == "denovo":
            run_tool_denovo(args.parent)
        elif args.tool_name == "trial":
            run_tool_clinical_simulation(args.drug, args.condition)
        else:
            tools_parser.print_help()
    elif args.command == "config":
        updates = {}
        if args.set_default_model:
            updates["OLLAMA_DEFAULT_MODEL"] = args.set_default_model
        if args.set_reasoning_model:
            updates["OLLAMA_REASONING_MODEL"] = args.set_reasoning_model
        if args.set_embedding_model:
            updates["OLLAMA_EMBEDDING_MODEL"] = args.set_embedding_model
        if args.set_depth:
            updates["DEFAULT_DEPTH"] = args.set_depth
        if args.set_url:
            updates["OLLAMA_BASE_URL"] = args.set_url

        if updates:
            update_settings(updates, persist=True)
            console.print("[bold green]✓ Configuration successfully updated and saved to .env:[/bold green]")
            for k, v in updates.items():
                console.print(f"  • {k} = [cyan]{v}[/cyan]")
        else:
            settings = get_settings()
            console.print(get_system_status_panel(client))
    elif args.command == "serve":
        start_server(host=args.host, port=args.port, reload=args.reload)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
