"""Server runner for HelixMind FastAPI application."""

from __future__ import annotations

import uvicorn
from rich.console import Console
from rich.panel import Panel

console = Console()


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Launch the HelixMind FastAPI & WebSocket server."""
    console.print(
        Panel.fit(
            f"[bold cyan]HelixMind API & Real-Time WebSockets Server[/bold cyan]\n"
            f"REST Endpoints: [bold green]http://{host}:{port}/api/health[/bold green]\n"
            f"Interactive OpenAPI Docs: [bold blue]http://{host}:{port}/docs[/bold blue]\n"
            f"Global WebSockets Stream: [bold magenta]ws://{host}:{port}/ws/live[/bold magenta]\n"
            f"Per-Investigation Stream: [bold magenta]ws://{host}:{port}/ws/investigations/{{id}}[/bold magenta]",
            title="API Gateway",
            border_style="cyan",
        )
    )
    uvicorn.run("helixmind.api.app:app", host=host, port=port, reload=reload)
