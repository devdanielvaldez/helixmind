"""WebSocket Connection Manager for real-time frontend broadcasting."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Set
from fastapi import WebSocket
from rich.console import Console

from helixmind.api.events import EventType, WebSocketEvent

console = Console()


class ConnectionManager:
    """Manages WebSocket connections and broadcasts events to frontend clients."""

    def __init__(self):
        # Maps investigation_id -> set of active WebSockets
        self.investigation_rooms: Dict[str, Set[WebSocket]] = {}
        # Global subscribers listening to all investigations
        self.global_subscribers: Set[WebSocket] = set()
        # Event history buffer per investigation for reconnecting clients
        self.event_history: Dict[str, List[Dict[str, Any]]] = {}
        # Active investigation state cache
        self.active_investigations: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, investigation_id: str = "all") -> None:
        """Accept a new WebSocket connection and register to a room."""
        await websocket.accept()
        if investigation_id == "all":
            self.global_subscribers.add(websocket)
        else:
            if investigation_id not in self.investigation_rooms:
                self.investigation_rooms[investigation_id] = set()
            self.investigation_rooms[investigation_id].add(websocket)

    def disconnect(self, websocket: WebSocket, investigation_id: str = "all") -> None:
        """Remove a disconnected WebSocket."""
        if investigation_id == "all":
            self.global_subscribers.discard(websocket)
        else:
            if investigation_id in self.investigation_rooms:
                self.investigation_rooms[investigation_id].discard(websocket)
                if not self.investigation_rooms[investigation_id]:
                    del self.investigation_rooms[investigation_id]

    async def broadcast_event(
        self,
        event_type: EventType,
        investigation_id: str,
        message: str = "",
        data: Dict[str, Any] | None = None,
    ) -> None:
        """Broadcast an event payload to both room subscribers and global listeners."""
        payload = WebSocketEvent(
            event=event_type,
            investigation_id=investigation_id,
            message=message,
            data=data or {},
        )
        json_data = payload.model_dump(mode="json")

        # Buffer event in memory for late reconnecting clients
        if investigation_id not in self.event_history:
            self.event_history[investigation_id] = []
        self.event_history[investigation_id].append(json_data)
        if len(self.event_history[investigation_id]) > 500:
            self.event_history[investigation_id] = self.event_history[investigation_id][-500:]

        # Update active investigation state cache
        if investigation_id != "all":
            if investigation_id not in self.active_investigations:
                self.active_investigations[investigation_id] = {
                    "investigation_id": investigation_id,
                    "status": "running",
                    "events_count": 0,
                    "last_message": message,
                    "papers": [],
                    "hypotheses": [],
                    "agents": [],
                    "milestones": [],
                    "graph": {"nodes_count": 0, "edges_count": 0},
                }
            st = self.active_investigations[investigation_id]
            st["events_count"] += 1
            st["last_message"] = message

            if data and "activity" in data:
                st["current_activity"] = {
                    "activity": data.get("activity"),
                    "message": message,
                    "title": data.get("title", ""),
                    "pmid": data.get("pmid"),
                    "paper": data.get("paper"),
                    "index": data.get("paper_index") or data.get("trial_index"),
                    "total": data.get("total_papers") or data.get("total_trials"),
                    "timestamp": payload.timestamp,
                }

            if event_type == EventType.INVESTIGATION_COMPLETED:
                st["status"] = "completed"
                st["current_activity"] = {
                    "activity": "completed",
                    "message": "Investigation completed. All artifacts saved.",
                    "timestamp": payload.timestamp,
                }
            elif event_type == EventType.INVESTIGATION_ERROR:
                st["status"] = "error"
            elif event_type == EventType.LITERATURE_INGESTED and data:
                if "papers" in data and isinstance(data["papers"], list):
                    st["papers"] = data["papers"]
            elif event_type == EventType.HYPOTHESIS_GENERATED and data:
                st["hypotheses"].append(data)
            elif event_type == EventType.AGENT_SPAWNED and data:
                st["agents"].append(data)
            elif event_type == EventType.ROADMAP_SYNTHESIZED and data:
                st["milestones"] = data.get("milestones", [])
            elif event_type == EventType.GRAPH_UPDATED and data:
                st["graph"] = data

        recipients: Set[WebSocket] = set(self.global_subscribers)
        if investigation_id in self.investigation_rooms:
            recipients.update(self.investigation_rooms[investigation_id])

        if not recipients:
            return

        dead_connections: List[WebSocket] = []
        for ws in recipients:
            try:
                await ws.send_json(json_data)
            except Exception:
                dead_connections.append(ws)

        for ws in dead_connections:
            self.global_subscribers.discard(ws)
            if investigation_id in self.investigation_rooms:
                self.investigation_rooms[investigation_id].discard(ws)

    def get_history(self, investigation_id: str) -> List[Dict[str, Any]]:
        """Return cached events for an investigation."""
        return self.event_history.get(investigation_id, [])

    def get_state(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """Return cached live state for an investigation."""
        return self.active_investigations.get(investigation_id)


# Global singleton instance
ws_manager = ConnectionManager()
