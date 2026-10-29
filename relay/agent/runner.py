"""Start, pause, resume and inspect agent runs.

A run is one pass of the graph over one ticket, keyed by run_id (also the LangGraph thread id).
The runner turns graph interrupts into an `awaiting_approval` status, resumes with a reviewer's
decision, and can continue a run that died mid-way from its last checkpoint.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from relay import sla
from relay.agent.graph import build_graph, graph_config, initial_state
from relay.agent.nodes import AgentDeps
from relay.approvals import ApprovalError, ApprovalStatus, Decision
from relay.domain import Ticket, TicketStatus, utcnow
from relay.memory import MemoryWriter


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class RunRecord(BaseModel):
    run_id: str
    ticket_id: str
    status: RunStatus = RunStatus.RUNNING
    outcome: dict[str, Any] | None = None
    pending_approval_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RunStore(Protocol):
    def save_run(self, run: RunRecord) -> RunRecord: ...

    def get_run(self, run_id: str) -> RunRecord | None: ...


class TicketStatusWriter(Protocol):
    def set_status(self, ticket_id: str, status: TicketStatus) -> None: ...


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def save_run(self, run: RunRecord) -> RunRecord:
        self._runs[run.run_id] = run
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list_runs(self, ticket_id: str | None = None) -> list[RunRecord]:
        return [r for r in self._runs.values() if ticket_id is None or r.ticket_id == ticket_id]


class AgentRunner:
    def __init__(
        self,
        deps: AgentDeps,
        *,
        checkpointer: Any = None,
        runs: RunStore | None = None,
        tickets: TicketStatusWriter | None = None,
        sla_policies: sla.SlaPolicyStore | None = None,
        memory_writer: MemoryWriter | None = None,
    ) -> None:
        self.deps = deps
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = build_graph(deps, self.checkpointer)
        self.runs = runs or InMemoryRunStore()
        self.tickets = tickets
        self.sla_policies = sla_policies or sla.StaticSlaPolicies()
        self.memory_writer = memory_writer

    def _config(self, run_id: str) -> dict[str, Any]:
        return graph_config(run_id, self.deps)

    async def start(self, ticket: Ticket, run_id: str | None = None) -> RunRecord:
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        deadline = sla.deadline(ticket, self.sla_policies.policies())
        record = self.runs.save_run(RunRecord(run_id=run_id, ticket_id=ticket.ticket_id))
        self._set_ticket(ticket.ticket_id, TicketStatus.IN_PROGRESS)
        state = initial_state(run_id, ticket.for_model(), deadline.isoformat())
        return await self._drive(record, state)

    async def resume(self, run_id: str, decision: Decision) -> RunRecord:
        record = self._require(run_id)
        if record.status != RunStatus.AWAITING_APPROVAL:
            raise ApprovalError(f"run {run_id} is {record.status}, not awaiting approval")
        return await self._drive(record, Command(resume=decision.model_dump()))

    async def decide(self, approval_id: str, decision: Decision) -> RunRecord:
        """Record a reviewer's decision and resume the run that was waiting on it."""
        approval = self.deps.approvals.get(approval_id)
        if approval is None:
            raise ApprovalError(f"unknown approval {approval_id}")
        if approval.status != ApprovalStatus.PENDING:
            raise ApprovalError(f"approval {approval_id} is already {approval.status}")
        self.deps.approvals.decide(approval_id, decision)
        return await self.resume(approval.run_id, decision)

    async def recover(self, run_id: str) -> RunRecord:
        """Continue a run from its last checkpoint, e.g. after the worker process was killed."""
        record = self._require(run_id)
        return await self._drive(record, None)

    async def snapshot(self, run_id: str) -> dict[str, Any]:
        state = await self.graph.aget_state(self._config(run_id))
        values = state.values or {}
        return {
            "run_id": run_id,
            "next": list(state.next),
            "plan": values.get("plan", []),
            "trajectory": values.get("trajectory", []),
            "outcome": values.get("outcome"),
            "tool_calls": values.get("tool_calls", 0),
        }

    def _require(self, run_id: str) -> RunRecord:
        record = self.runs.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def _set_ticket(self, ticket_id: str, status: TicketStatus) -> None:
        if self.tickets is not None:
            self.tickets.set_status(ticket_id, status)

    async def _drive(self, record: RunRecord, payload: Any) -> RunRecord:
        config = self._config(record.run_id)
        try:
            await self.graph.ainvoke(payload, config)
        except Exception as exc:
            failed = record.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at": utcnow(),
                }
            )
            self.runs.save_run(failed)
            raise
        state = await self.graph.aget_state(config)
        values = state.values or {}
        if state.next:
            approval = values.get("approval") or {}
            update = {
                "status": RunStatus.AWAITING_APPROVAL,
                "pending_approval_id": approval.get("approval_id"),
            }
            ticket_status = TicketStatus.AWAITING_APPROVAL
        else:
            outcome = values.get("outcome") or {}
            update = {
                "status": RunStatus.COMPLETED,
                "outcome": outcome,
                "pending_approval_id": None,
            }
            ticket_status = (
                TicketStatus.RESOLVED
                if outcome.get("status") == "resolved"
                else TicketStatus.ESCALATED
            )
        saved = self.runs.save_run(
            record.model_copy(update={**update, "error": None, "updated_at": utcnow()})
        )
        self._set_ticket(record.ticket_id, ticket_status)
        if ticket_status == TicketStatus.RESOLVED and self.memory_writer is not None:
            await self.memory_writer.remember(values["ticket"], saved.outcome.get("reply") or "")
        return saved
