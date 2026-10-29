"""HTTP API: create tickets, run the agent, review approvals and work escalations.

.venv/bin/python -m uvicorn relay.api.app:create_app --factory --port 8000
"""

from __future__ import annotations

import json
import tempfile
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from relay import sla
from relay.agent.runner import RunRecord
from relay.api.services import Services, build_services
from relay.approvals import Approval, ApprovalError, ApprovalStatus, Decision
from relay.config import Settings, get_settings
from relay.domain import Customer, EscalationReason, Priority, Ticket, TicketCreate, utcnow
from relay.escalation import Escalation, EscalationStatus, Resolution, to_eval_task
from relay.providers.base import MissingDependencyError
from relay.tracing import setup_tracing
from relay.voice import voicemail_ticket


class RunView(BaseModel):
    run: RunRecord
    plan: list[str]
    trajectory: list[dict[str, Any]]
    outcome: dict[str, Any] | None


ServicesFactory = Callable[[Settings], Any]


def create_app(
    settings: Settings | None = None, services_factory: ServicesFactory | None = None
) -> FastAPI:
    settings = settings or get_settings()
    factory = services_factory or (lambda s: build_services(s))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_tracing(settings)
        async with factory(settings) as services:
            app.state.services = services
            yield

    app = FastAPI(title="Relay", version="0.1.0", lifespan=lifespan)

    def services(request: Request) -> Services:
        return request.app.state.services

    def ticket_or_404(svc: Services, ticket_id: str) -> Ticket:
        ticket = svc.tickets.get_ticket(ticket_id)
        if ticket is None:
            raise HTTPException(404, f"ticket {ticket_id} not found")
        return ticket

    def create_ticket_record(svc: Services, body: TicketCreate) -> Ticket:
        if svc.tickets.get_customer(body.customer_id) is None:
            svc.tickets.put_customer(Customer(customer_id=body.customer_id, name=body.customer_id))
        ticket = Ticket(ticket_id=f"T-{uuid.uuid4().hex[:8]}", **body.model_dump())
        svc.tickets.put_ticket(ticket)
        return ticket

    @app.get("/health")
    async def health(svc: Services = Depends(services)) -> dict[str, Any]:
        return {
            "status": "ok",
            **svc.provider.describe(),
            "storage": svc.storage,
            "checkpointer": svc.settings.checkpointer,
            "approval_tools": svc.settings.approval_tools,
        }

    # Tickets ---------------------------------------------------------------------------

    @app.post("/tickets", status_code=201)
    async def create_ticket(body: TicketCreate, svc: Services = Depends(services)) -> Ticket:
        return create_ticket_record(svc, body)

    @app.post("/tickets/voicemail", status_code=201)
    async def create_voicemail_ticket(
        customer_id: str = Form(...),
        priority: Priority = Form(Priority.NORMAL),
        audio: UploadFile = File(...),
        svc: Services = Depends(services),
    ) -> Ticket:
        suffix = Path(audio.filename or "voicemail.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp.flush()
            try:
                transcript = svc.transcriber.transcribe(Path(tmp.name))
            except MissingDependencyError as exc:
                raise HTTPException(501, str(exc)) from exc
        try:
            body = voicemail_ticket(transcript, customer_id, priority)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return create_ticket_record(svc, body)

    @app.get("/tickets/queue")
    async def ticket_queue(svc: Services = Depends(services)) -> list[sla.SlaStatus]:
        return sla.rank(svc.tickets.list_tickets(), svc.sla_policies.policies(), utcnow())

    @app.get("/tickets/{ticket_id}")
    async def get_ticket(ticket_id: str, svc: Services = Depends(services)) -> Ticket:
        return ticket_or_404(svc, ticket_id)

    @app.post("/tickets/{ticket_id}/run", status_code=201)
    async def run_ticket(ticket_id: str, svc: Services = Depends(services)) -> RunView:
        ticket = ticket_or_404(svc, ticket_id)
        run = await svc.runner.start(ticket)
        return await run_view(svc, run.run_id)

    # Runs ------------------------------------------------------------------------------

    async def run_view(svc: Services, run_id: str) -> RunView:
        run = svc.runs.get_run(run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        snap = await svc.runner.snapshot(run_id)
        return RunView(
            run=run, plan=snap["plan"], trajectory=snap["trajectory"], outcome=snap["outcome"]
        )

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, svc: Services = Depends(services)) -> RunView:
        return await run_view(svc, run_id)

    @app.post("/runs/{run_id}/recover")
    async def recover_run(run_id: str, svc: Services = Depends(services)) -> RunView:
        if svc.runs.get_run(run_id) is None:
            raise HTTPException(404, f"run {run_id} not found")
        await svc.runner.recover(run_id)
        return await run_view(svc, run_id)

    # Approvals -------------------------------------------------------------------------

    @app.get("/approvals")
    async def list_approvals(
        status: ApprovalStatus | None = ApprovalStatus.PENDING, svc: Services = Depends(services)
    ) -> list[Approval]:
        return svc.approvals.list(status)

    @app.post("/approvals/{approval_id}")
    async def decide_approval(
        approval_id: str, decision: Decision, svc: Services = Depends(services)
    ) -> RunView:
        approval = svc.approvals.get(approval_id)
        if approval is None:
            raise HTTPException(404, f"approval {approval_id} not found")
        try:
            run = await svc.runner.decide(approval_id, decision)
        except ApprovalError as exc:
            raise HTTPException(409, str(exc)) from exc
        return await run_view(svc, run.run_id)

    # Escalations -----------------------------------------------------------------------

    @app.get("/escalations")
    async def list_escalations(
        status: EscalationStatus | None = None, svc: Services = Depends(services)
    ) -> list[Escalation]:
        return svc.escalations.store.list(status)

    @app.post("/escalations/{escalation_id}/resolve")
    async def resolve_escalation(
        escalation_id: str, resolution: Resolution, svc: Services = Depends(services)
    ) -> Escalation:
        try:
            escalation = svc.escalations.resolve(escalation_id, resolution)
        except KeyError as exc:
            raise HTTPException(404, f"escalation {escalation_id} not found") from exc
        ticket = svc.tickets.get_ticket(escalation.ticket_id)
        if ticket is not None:
            await svc.memory_writer.remember(ticket.for_model(), resolution.resolution)
        return escalation

    @app.get("/escalations/export", response_class=PlainTextResponse)
    async def export_escalations(svc: Services = Depends(services)) -> str:
        """Resolved escalations as eval tasks, one JSON object per line."""
        lines = []
        for escalation in svc.escalations.store.list(EscalationStatus.RESOLVED):
            ticket = svc.tickets.get_ticket(escalation.ticket_id)
            if ticket is not None:
                lines.append(json.dumps(to_eval_task(escalation, ticket)))
        return "\n".join(lines) + ("\n" if lines else "")

    @app.post("/sla/sweep")
    async def sla_sweep(svc: Services = Depends(services)) -> list[dict[str, Any]]:
        """Escalate open tickets whose SLA has run out and that no run is working on."""
        from relay.domain import TicketStatus

        swept = []
        for status in sla.breached(
            svc.tickets.list_tickets(), svc.sla_policies.policies(), utcnow()
        ):
            ticket = svc.tickets.get_ticket(status.ticket_id)
            if ticket is None or ticket.status != TicketStatus.OPEN:
                continue
            record = svc.escalations.escalate(
                ticket.ticket_id,
                EscalationReason.SLA_BREACH.value,
                f"SLA deadline {status.deadline.isoformat()} passed before any agent run.",
                None,
                [],
            )
            svc.tickets.set_status(ticket.ticket_id, TicketStatus.ESCALATED)
            swept.append(record)
        return swept

    return app
