"""Event models and schemas for real-time WebSocket communication."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of real-time WebSocket events broadcast during research."""
    INVESTIGATION_STARTED = "investigation:started"
    AGENT_SPAWNED = "agent:spawned"
    LITERATURE_INGESTED = "literature:ingested"
    EVIDENCE_EXTRACTED = "evidence:extracted"
    GRAPH_UPDATED = "graph:updated"
    DIRECTOR_CYCLE = "director:cycle"
    HYPOTHESIS_GENERATED = "hypothesis:generated"
    HYPOTHESIS_CRITIQUED = "hypothesis:critiqued"
    ROADMAP_SYNTHESIZED = "roadmap:synthesized"
    PAPER_AUTHORED = "paper:authored"
    INVESTIGATION_COMPLETED = "investigation:completed"
    INVESTIGATION_ERROR = "investigation:error"
    STATUS_LOG = "status:log"


class WebSocketEvent(BaseModel):
    """Payload sent across WebSocket connections to frontend subscribers."""
    event: EventType
    investigation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
