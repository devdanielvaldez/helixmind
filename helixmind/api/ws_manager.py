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


# Global singleton instance
ws_manager = ConnectionManager()
