"""Unit tests for FastAPI REST API and real-time WebSockets."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import pytest
from helixmind.api.app import app
from helixmind.api.events import EventType
from helixmind.api.ws_manager import ws_manager

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "available_models" in data


def test_api_list_investigations():
    response = client.get("/api/investigations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_list_papers():
    response = client.get("/api/papers")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_post_investigation():
    payload = {
        "topic": "test condition",
        "max_papers": 1,
        "max_trials": 0,
        "depth": "quick",
    }
    with patch("helixmind.autopilot.orchestrator.AutoPilotOrchestrator.investigate_async", new_callable=AsyncMock) as mock_inv:
        response = client.post("/api/investigations", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "investigation_id" in data
        assert data["investigation_id"].startswith("INV-")
        assert "/ws/investigations/" in data["websocket_url"]


def test_websocket_global_channel():
    with client.websocket_connect("/ws/live") as websocket:
        msg = websocket.receive_json()
        assert msg["event"] == EventType.STATUS_LOG.value
        assert "Connected to global" in msg["message"]


@pytest.mark.asyncio
async def test_websocket_manager_broadcast():
    inv_id = "INV-TEST-WS"
    # Verify manager handles broadcasting without crashing even with empty room
    await ws_manager.broadcast_event(
        event_type=EventType.STATUS_LOG,
        investigation_id=inv_id,
        message="Test event",
        data={"key": "val"},
    )
