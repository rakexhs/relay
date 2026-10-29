from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from relay.api.app import create_app
from relay.api.services import build_services
from relay.config import Settings


class FakeTranscriber:
    def __init__(self, text):
        self.text = text
        self.paths = []

    def transcribe(self, audio_path: Path) -> str:
        self.paths.append(audio_path)
        return self.text


@pytest.fixture
def client():
    settings = Settings(_env_file=None)
    transcriber = FakeTranscriber(
        "Hi, this is Sam. Since this morning every API call returns 401 after we rotated our "
        "API keys. Please call me back."
    )

    @asynccontextmanager
    async def factory(s):
        async with build_services(s, transcriber=transcriber) as services:
            yield services

    with TestClient(create_app(settings, factory)) as c:
        yield c


def _ticket(client, **overrides):
    body = {
        "customer_id": "C-1001",
        "subject": "Webhooks failing with 502",
        "body": "Our webhook endpoint stopped receiving events, we see 502 errors in the dashboard.",
        "priority": "high",
        **overrides,
    }
    response = client.post("/tickets", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "stub"
    assert body["storage"] == "memory"


def test_ticket_run_approve_resolve(client):
    ticket = _ticket(client)
    run = client.post(f"/tickets/{ticket['ticket_id']}/run").json()
    assert run["run"]["status"] == "awaiting_approval"
    assert run["trajectory"][-1]["type"] == "approval_requested"

    pending = client.get("/approvals").json()
    assert len(pending) == 1 and pending[0]["tool"] == "draft_reply"
    assert pending[0]["args"]["send"] is True

    decided = client.post(
        f"/approvals/{pending[0]['approval_id']}", json={"approved": True, "reviewer": "kim"}
    ).json()
    assert decided["run"]["status"] == "completed"
    assert decided["outcome"]["status"] == "resolved"
    assert "req_7f2a" in decided["outcome"]["reply"]

    again = client.post(f"/approvals/{pending[0]['approval_id']}", json={"approved": True})
    assert again.status_code == 409

    view = client.get(f"/runs/{run['run']['run_id']}").json()
    tools = [s["tool"] for s in view["trajectory"] if s["type"] == "tool_call"]
    assert tools[0] == "recall_customer_history" and tools[-1] == "draft_reply"
    assert client.get(f"/tickets/{ticket['ticket_id']}").json()["status"] == "resolved"


def test_escalation_resolve_and_export(client):
    ticket = _ticket(
        client,
        subject="Refund please",
        body="We were charged twice for March and would like a refund of the duplicate.",
        priority="normal",
    )
    run = client.post(f"/tickets/{ticket['ticket_id']}/run").json()
    approval_id = run["run"]["pending_approval_id"]
    done = client.post(f"/approvals/{approval_id}", json={"approved": True}).json()
    assert done["outcome"]["reason"] == "needs_refund"

    escalations = client.get("/escalations").json()
    assert escalations[0]["reason"] == "needs_refund"
    assert escalations[0]["trajectory"]
    resolved = client.post(
        f"/escalations/{escalations[0]['escalation_id']}/resolve",
        json={
            "resolution": "Billing refunded the duplicate charge.",
            "resolved_by": "kim",
            "agent_should": "escalate",
        },
    )
    assert resolved.json()["status"] == "resolved"
    lines = client.get("/escalations/export").text.strip().splitlines()
    assert len(lines) == 1
    assert '"outcome": "escalated"' in lines[0]


def test_voicemail_becomes_a_ticket_on_the_same_path(client):
    response = client.post(
        "/tickets/voicemail",
        data={"customer_id": "C-1002", "priority": "high"},
        files={"audio": ("vm.wav", b"RIFF....WAVEfmt ", "audio/wav")},
    )
    assert response.status_code == 201, response.text
    ticket = response.json()
    assert ticket["channel"] == "voicemail"
    assert ticket["subject"].startswith("Voicemail: Hi, this is Sam.")
    run = client.post(f"/tickets/{ticket['ticket_id']}/run").json()
    docs = next(s for s in run["trajectory"] if s.get("tool") == "search_docs")
    assert docs["result"]["results"][0]["doc_id"] == "KB-104"


def test_queue_and_404s(client):
    _ticket(client, priority="urgent", severity="critical")
    queue = client.get("/tickets/queue").json()
    assert queue[0]["priority"] == "urgent"
    assert client.get("/tickets/T-nope").status_code == 404
    assert client.get("/runs/run-nope").status_code == 404
    assert client.post("/approvals/A-nope", json={"approved": True}).status_code == 404
    assert (
        client.post(
            "/tickets", json={"customer_id": "bad", "subject": "s", "body": "b"}
        ).status_code
        == 422
    )


def test_api_can_reach_tools_over_mcp():
    settings = Settings(_env_file=None, tool_transport="mcp")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["tool_transport"] == "mcp"
        ticket = _ticket(c)
        run = c.post(f"/tickets/{ticket['ticket_id']}/run").json()
        done = c.post(f"/approvals/{run['run']['pending_approval_id']}", json={"approved": True})
        assert done.json()["outcome"]["status"] == "resolved"
