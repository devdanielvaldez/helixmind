"""FastAPI Web Application providing REST endpoints and real-time WebSockets."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from helixmind import __version__
from helixmind.api.events import EventType
from helixmind.api.ws_manager import ws_manager
from helixmind.config import get_settings
from helixmind.copilot import CopilotResponse, ResearchCopilot
from helixmind.insilico import (
    BioVisualizer,
    CRISPRDesignReport,
    CRISPRDesigner,
    ChemoinformaticsEngine,
    ClinicalSimulator,
    ClinVarStratifier,
    DeNovoGenerator,
    DepMapVerifier,
    DockingSimulator,
    FDAPharmacovigilance,
    StructureRetriever,
    TMEMapper,
)
from helixmind.llm.ollama_client import OllamaClient
from helixmind.papers import PDFParser
from helixmind.publishing import BibTeXExporter, LaTeXExporter, PDFExporter

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
    depth: str = Field(default="deep", description="deep (2 cycles, default), exhaustive (3 cycles), or institutional (4 cycles)")
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

    from helixmind.autopilot.orchestrator import AutoPilotOrchestrator
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


@app.get("/api/papers/{filename}/pdf")
async def download_paper_pdf(filename: str):
    """Render and download academic PDF monograph formatted for Nature/Cell."""
    paper_path = settings.data_dir / "papers" / filename
    if not paper_path.exists():
        raise HTTPException(status_code=404, detail="Paper not found")

    pdf_filename = filename.replace(".md", ".pdf")
    pdf_path = settings.data_dir / "papers" / pdf_filename

    # Build PDF using ReportLab
    markdown_content = paper_path.read_text(encoding="utf-8")
    PDFExporter.export_pdf(markdown_content, pdf_path)

    return FileResponse(
        path=str(pdf_path),
        filename=pdf_filename,
        media_type="application/pdf",
    )


@app.get("/api/papers/{filename}/latex")
async def download_paper_latex(filename: str):
    """Export academic LaTeX (.tex) two-column source file."""
    paper_path = settings.data_dir / "papers" / filename
    if not paper_path.exists():
        raise HTTPException(status_code=404, detail="Paper not found")

    tex_filename = filename.replace(".md", ".tex")
    tex_path = settings.data_dir / "papers" / tex_filename

    markdown_content = paper_path.read_text(encoding="utf-8")
    LaTeXExporter.export_latex(markdown_content, tex_path)

    return FileResponse(
        path=str(tex_path),
        filename=tex_filename,
        media_type="application/x-tex",
    )


# -----------------------------------------------------------------------------
# In-Silico & Chemoinformatics Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/insilico/molecule/{name_or_smiles}")
async def get_molecular_properties(name_or_smiles: str) -> Dict[str, Any]:
    """Compute Lipinski Rule of 5, LogP, TPSA, and druglikeness using RDKit."""
    props = ChemoinformaticsEngine.analyze_molecule(name_or_smiles)
    if not props:
        raise HTTPException(status_code=404, detail=f"Could not compute properties for '{name_or_smiles}'")
    return props.model_dump()


@app.get("/api/insilico/docking")
async def simulate_docking(ligand: str = "tamoxifen", target: str = "HER2") -> Dict[str, Any]:
    """Simulate in-silico ligand-receptor binding affinity ΔG and Ki."""
    result = DockingSimulator.predict_docking_affinity(ligand, target)
    return result.model_dump()


@app.get("/api/insilico/structure/{target_name}")
async def get_target_structure(target_name: str) -> Dict[str, Any]:
    """Fetch crystallographic 3D PDB structure metadata for a receptor."""
    retriever = StructureRetriever()
    structure = await retriever.fetch_structure_async(target_name)
    if not structure:
        raise HTTPException(status_code=404, detail=f"No PDB structure found for '{target_name}'")
    return structure.model_dump()


@app.get("/api/insilico/depmap/{gene_symbol}")
async def verify_gene_depmap(gene_symbol: str, lineage: str = "") -> Dict[str, Any]:
    """Query Broad Institute DepMap CRISPR essentiality and Chronos gene effect scores."""
    verifier = DepMapVerifier()
    res = await verifier.verify_gene_essentiality_async(gene_symbol, target_condition=lineage)
    return res.model_dump()


class DeNovoRequest(BaseModel):
    parent_drug: str = "tamoxifen"
    target_goal: str = "Optimize affinity and BBB permeability"
    max_analogs: int = 3


@app.post("/api/insilico/denovo")
async def generate_denovo_analogs(req: DeNovoRequest) -> List[Dict[str, Any]]:
    """Generate bioisosterically optimized chemical analogs with SAscore synthesizability and ADMET."""
    analogs = DeNovoGenerator.generate_optimized_analogs(
        parent_drug_name=req.parent_drug,
        target_biological_goal=req.target_goal,
        max_analogs=req.max_analogs,
    )
    return [a.model_dump() for a in analogs]


class ClinicalSimRequest(BaseModel):
    drug_name: str = "Lead Molecule"
    target_condition: str = "glioblastoma"
    delta_g: float = -8.5
    cohort_size: int = 200


@app.post("/api/insilico/clinical-simulation")
async def simulate_clinical_trial_endpoint(req: ClinicalSimRequest) -> Dict[str, Any]:
    """Run an in-silico double-blind randomized clinical trial in virtual patient cohorts (Digital Twins)."""
    report = ClinicalSimulator.simulate_trial(
        drug_name=req.drug_name,
        target_condition=req.target_condition,
        predicted_delta_g=req.delta_g,
        cohort_size=req.cohort_size,
    )
    return report.model_dump()


@app.get("/api/insilico/fda/{drug_name}")
async def get_fda_pharmacovigilance(drug_name: str, limit: int = 10) -> Dict[str, Any]:
    """Retrieve real human post-marketing adverse reactions and safety signals from openFDA FAERS."""
    fda_client = FDAPharmacovigilance()
    report = await fda_client.fetch_safety_profile_async(drug_name, limit=limit)
    return report.model_dump()


@app.get("/api/insilico/clinvar/{gene_symbol}")
async def get_clinvar_stratification(gene_symbol: str, condition: str = "Oncology") -> Dict[str, Any]:
    """Retrieve pathogenic variants and precision inclusion criteria from NCBI ClinVar."""
    stratifier = ClinVarStratifier()
    report = await stratifier.stratify_target_async(gene_symbol, target_condition=condition)
    return report.model_dump()


@app.get("/api/insilico/tme/map")
async def get_tme_map(targets: str = "CD274,STAT3,TGFB1", condition: str = "Solid Tumor") -> Dict[str, Any]:
    """Deconvolute single-cell tumor microenvironment and map paracrine ligand-receptor axes."""
    gene_list = [g.strip() for g in targets.split(",") if g.strip()]
    report = TMEMapper.analyze_microenvironment(gene_list, condition=condition)
    return report.model_dump()


# -----------------------------------------------------------------------------
# Document & PDF Ingestion Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/documents/upload")
async def upload_pdf_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and parse a local laboratory PDF manuscript or clinical protocol."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_path = settings.data_dir / "uploads" / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    parser = PDFParser()
    paper = parser.parse_pdf_file(upload_path, original_filename=file.filename)
    chunks = parser.extract_chunks(paper)

    return {
        "status": "success",
        "filename": file.filename,
        "paper_id": paper.id,
        "title": paper.title,
        "abstract": paper.abstract[:300] + "...",
        "authors": paper.authors,
        "extracted_chunks_count": len(chunks),
        "filepath": str(upload_path),
    }


# -----------------------------------------------------------------------------
# CRISPR sgRNA & PCR Primer Design Endpoints
# -----------------------------------------------------------------------------

class CRISPRRequest(BaseModel):
    gene_name: str
    fallback_dna: Optional[str] = None


@app.post("/api/crispr/design", response_model=CRISPRDesignReport)
async def design_crispr(req: CRISPRRequest) -> CRISPRDesignReport:
    """Design CRISPR-Cas9 sgRNAs, PAM sites, cloning oligos, and PCR validation primers."""
    designer = CRISPRDesigner()
    return await designer.design_for_gene_async(req.gene_name, fallback_dna=req.fallback_dna)


# -----------------------------------------------------------------------------
# DNA & Molecular 2D Vector Graphics (SVG)
# -----------------------------------------------------------------------------

@app.get("/api/visuals/molecule/{name_or_smiles}")
async def get_molecule_svg(name_or_smiles: str):
    """Render publication-grade 2D vector chemical structure in SVG using RDKit."""
    svg = BioVisualizer.render_molecule_svg(name_or_smiles)
    if not svg:
        raise HTTPException(status_code=404, detail=f"Could not render chemical structure for '{name_or_smiles}'")
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/visuals/dna")
async def get_dna_construct_svg(construct_name: str = "pCas9-sgRNA-Pur"):
    """Render a linear genetic cassette or plasmid vector in SVG."""
    svg = BioVisualizer.render_dna_construct_svg(construct_name=construct_name)
    return Response(content=svg, media_type="image/svg+xml")


# -----------------------------------------------------------------------------
# Interactive Research Copilot Dialogue Endpoints
# -----------------------------------------------------------------------------

class CopilotChatRequest(BaseModel):
    message: str
    topic: str = ""


@app.post("/api/investigations/{inv_id}/chat", response_model=CopilotResponse)
async def chat_with_copilot(inv_id: str, req: CopilotChatRequest) -> CopilotResponse:
    """Send a real-time question or steering instruction to the investigation copilot."""
    copilot = ResearchCopilot()
    return await copilot.chat_async(
        investigation_id=inv_id,
        user_message=req.message,
        topic=req.topic,
    )


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


@app.websocket("/ws/chat/{inv_id}")
async def websocket_copilot_chat(websocket: WebSocket, inv_id: str):
    """Bidirectional real-time WebSocket chat with the investigation copilot."""
    await websocket.accept()
    copilot = ResearchCopilot()
    try:
        await websocket.send_json({
            "sender": "copilot",
            "reply": f"Connected to live copilot for investigation {inv_id}. Ask questions or give research directions.",
            "citations": [],
            "actions_taken": [],
        })
        while True:
            data = await websocket.receive_json()
            user_msg = data.get("message", "")
            topic = data.get("topic", "")
            if user_msg.strip():
                response = await copilot.chat_async(
                    investigation_id=inv_id,
                    user_message=user_msg,
                    topic=topic,
                )
                await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# Mount Built Interactive War Room Frontend
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
