"""HelixMind Command Line Interface.

Provides diagnostics, model status, and live verification with Ollama.
"""

from __future__ import annotations

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from helixmind import __version__
from helixmind.api import start_server
from helixmind.autopilot import AutoPilotOrchestrator
from helixmind.config import get_settings
from helixmind.hypotheses.models import ResearchReport
from helixmind.llm.ollama_client import OllamaClient, OllamaConnectionError

console = Console()


def cmd_investigate(
    topic: str,
    max_papers: int = 5,
    max_trials: int = 4,
    depth: str = "deep",
    model: str | None = None,
) -> None:
    """Launch an autonomous multi-agent scientific investigation."""
    import asyncio

    orchestrator = AutoPilotOrchestrator()
    console.print(
        Panel.fit(
            f"[bold green]Starting HelixMind Autonomous Investigation[/bold green]\n"
            f"Topic: [bold yellow]{topic}[/bold yellow]\n"
            f"Depth Level: [bold cyan]{depth.upper()}[/bold cyan] | Target Papers: [cyan]{max_papers}[/cyan] | Clinical Trials: [cyan]{max_trials}[/cyan]"
            + (f" | Model: [magenta]{model}[/magenta]" if model else ""),
            title="Auto-Pilot Research Engine",
            border_style="green",
        )
    )

    report: ResearchReport = asyncio.run(
        orchestrator.investigate_async(
            topic=topic,
            max_papers=max_papers,
            max_trials=max_trials,
            depth=depth,
            model=model,
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
                f"Format: [white]GitHub Markdown (Abstract, Terminology, Intro, Mermaid Knowledge Graph, Hypotheses, Roadmap, References)[/white]",
                title="Scientific Paper Output",
                border_style="magenta",
            )
        )



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
        sys.exit(1)

    console.print("[bold green]✓ Ollama server is active and reachable![/bold green]\n")

    try:
        models = client.list_models()
    except Exception as e:
        console.print(f"[bold red]Error fetching models:[/bold red] {e}")
        sys.exit(1)

    if not models:
        console.print("[yellow]No models found in Ollama. Pull one with `ollama pull llama3.2:3b`.[/yellow]")
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
        sys.exit(1)


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
            console.print(f"[bold green]✓ Embedding generated successfully![/bold green]")
            console.print(f"Dimension: [bold]{len(vec)}[/bold] floats")
            preview = ", ".join(f"{v:.4f}" for v in vec[:5]) + f", ... (total {len(vec)})"
            console.print(f"Vector preview: [dim][{preview}][/dim]\n")
        else:
            console.print("[bold red]✗ No embedding returned.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error during embedding generation:[/bold red] {e}")
        sys.exit(1)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="HelixMind Scientific Research Platform CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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
        default=5,
        help="Max papers to fetch and index (default: 5)",
    )
    inv_parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Max clinical trials to fetch (default: 3)",
    )
    inv_parser.add_argument(
        "--depth",
        type=str,
        choices=["deep", "exhaustive", "institutional"],
        default="deep",
        help="Depth of autonomous investigation: deep (2 cycles, default), exhaustive (3 cycles), institutional (4 cycles)",
    )
    inv_parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="LLM model override (e.g. 'llama3.2:1b' for fast CPU inference or 'llama3.2:3b')",
    )

    # Command: serve (Launch FastAPI & WebSockets gateway)
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST & real-time WebSockets API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable hot reloading")

    args = parser.parse_args()
    client = OllamaClient()

    if args.command == "status" or args.command is None:
        cmd_status(client)
    elif args.command == "chat":
        cmd_test_chat(client, prompt=args.prompt, model=args.model)
    elif args.command == "embed":
        cmd_test_embed(client, sample_text=args.text, model=args.model)
    elif args.command == "investigate":
        cmd_investigate(
            topic=args.topic,
            max_papers=args.papers,
            max_trials=args.trials,
            depth=args.depth,
            model=args.model,
        )
    elif args.command == "serve":
        start_server(host=args.host, port=args.port, reload=args.reload)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
