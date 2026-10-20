"""FastAPI Web Application providing REST endpoints and real-time WebSockets."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from helixmind import __version__
from helixmind.api.events import EventType
from helixmind.api.ws_manager import ws_manager
from helixmind.autopilot.orchestrator import AutoPilotOrchestrator
from helixmind.config import get_settings
from helixmind.llm.ollama_client import OllamaClient

app = FastAPI(
    title="HelixMind API",
    description="Autonomous Scientific Research Platform with Real-Time WebSockets",
    version=__version__,
)

# Enable CORS for frontend frameworks (React, Next.js, Vue, Svelte) on any local/remote port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
ollama = OllamaClient()


class InvestigationRequest(BaseModel):
    """Payload to launch a new autonomous scientific research mission."""
    topic: str = Field(description="Target disease, cellular mechanism, or biological topic to investigate")
    max_papers: int = Field(default=4, ge=1, le=20, description="Max literature to ingest")
    max_trials: int = Field(default=2, ge=0, le=10, description="Max clinical trials to analyze")
    depth: str = Field(default="standard", description="quick, standard, deep, or exhaustive")
    model: Optional[str] = Field(default=None, description="Model override (e.g. llama3.2:1b)")


class InvestigationStatusResponse(BaseModel):
    """Response returned upon launching an investigation."""
    investigation_id: str
    status: str = "running"
    topic: str
    depth: str
    started_at: str
    websocket_url: str


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
async def get_health() -> Dict[str, Any]:
    """Health check verifying API, Ollama server, and installed models."""
    ollama_ok = await ollama.check_health_async()
    models = await ollama.list_models_async() if ollama_ok else []
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "version": __version__,
        "ollama_connected": ollama_ok,
        "default_model": settings.ollama_default_model,
        "embedding_model": settings.ollama_embedding_model,
        "models_count": len(models),
        "available_models": [m.get("name") for m in models],
    }


@app.post("/api/investigations", response_model=InvestigationStatusResponse)
async def launch_investigation(
    req: InvestigationRequest,
    background_tasks: BackgroundTasks,
) -> InvestigationStatusResponse:
    """Launch an autonomous scientific investigation in the background."""
    inv_id = f"INV-{int(datetime.now().timestamp())}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    orchestrator = AutoPilotOrchestrator(settings=settings, ollama_client=ollama)

    # Launch autonomous orchestration in the background
    async def run_in_background():
        try:
            await orchestrator.investigate_async(
                topic=req.topic,
                max_papers=req.max_papers,
                max_trials=req.max_trials,
                depth=req.depth,
                model=req.model,
            )
        except Exception as ex:
            await ws_manager.broadcast_event(
                event_type=EventType.INVESTIGATION_ERROR,
                investigation_id=inv_id,
                message=f"Investigation failed: {ex}",
                data={"error": str(ex)},
            )

    background_tasks.add_task(run_in_background)

    return InvestigationStatusResponse(
        investigation_id=inv_id,
        status="running",
        topic=req.topic,
        depth=req.depth,
        started_at=now_str,
        websocket_url=f"/ws/investigations/{inv_id}",
    )


@app.get("/api/investigations")
async def list_investigations() -> List[Dict[str, Any]]:
    """List all previous investigations and reports."""
    reports_dir = settings.data_dir / "reports"
    if not reports_dir.exists():
        return []

    investigations: List[Dict[str, Any]] = []
    for f in sorted(reports_dir.glob("report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as rf:
                data = json.load(rf)
                investigations.append({
                    "filename": f.name,
                    "topic": data.get("topic"),
                    "timestamp": data.get("timestamp"),
                    "papers_analyzed": data.get("papers_analyzed", 0),
                    "clinical_trials_analyzed": data.get("clinical_trials_analyzed", 0),
                    "hypotheses_count": len(data.get("hypotheses", [])),
                    "paper_markdown_path": data.get("paper_markdown_path"),
                })
        except Exception:
            continue

    return investigations


@app.get("/api/investigations/{inv_slug}")
async def get_investigation_report(inv_slug: str) -> Dict[str, Any]:
    """Retrieve full JSON report by slug or ID."""
    reports_dir = settings.data_dir / "reports"
    # Search for matching file
    for f in reports_dir.glob("report_*.json"):
        if inv_slug in f.name:
            with open(f, "r", encoding="utf-8") as rf:
                return json.load(rf)

    raise HTTPException(status_code=404, detail="Investigation report not found")


@app.get("/api/papers")
async def list_papers() -> List[Dict[str, Any]]:
    """List all publication-grade scientific papers authored in Markdown."""
    papers_dir = settings.data_dir / "papers"
    if not papers_dir.exists():
        return []

    results: List[Dict[str, Any]] = []
    for f in sorted(papers_dir.glob("paper_*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        results.append({
            "filename": f.name,
            "filepath": str(f),
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })

    return results


@app.get("/api/papers/{filename}")
async def get_paper_content(filename: str) -> Dict[str, Any]:
    """Retrieve the raw GitHub Markdown content of an authored scientific paper."""
    paper_path = settings.data_dir / "papers" / filename
    if not paper_path.exists():
        raise HTTPException(status_code=404, detail="Paper not found")

    content = paper_path.read_text(encoding="utf-8")
    return {
        "filename": filename,
        "markdown": content,
    }


# -----------------------------------------------------------------------------
# WebSockets Real-Time Channel
# -----------------------------------------------------------------------------

@app.websocket("/ws/investigations/{inv_id}")
async def websocket_investigation_endpoint(websocket: WebSocket, inv_id: str):
    """Real-time WebSocket streaming all events for a specific investigation."""
    await ws_manager.connect(websocket, investigation_id=inv_id)
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "event": EventType.STATUS_LOG,
            "investigation_id": inv_id,
            "message": f"Connected to real-time research feed for {inv_id}",
            "data": {},
        })
        while True:
            # Keep socket alive and receive client pings/messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, investigation_id=inv_id)
    except Exception:
        ws_manager.disconnect(websocket, investigation_id=inv_id)


@app.websocket("/ws/live")
async def websocket_global_endpoint(websocket: WebSocket):
    """Global real-time WebSocket streaming all activities across all investigations."""
    await ws_manager.connect(websocket, investigation_id="all")
    try:
        await websocket.send_json({
            "event": EventType.STATUS_LOG,
            "investigation_id": "all",
            "message": "Connected to global HelixMind research broadcast stream",
            "data": {},
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, investigation_id="all")
    except Exception:
        ws_manager.disconnect(websocket, investigation_id="all")
