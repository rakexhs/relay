"""Escalations: hand-offs to a human, with the agent's trajectory and a reason code.

When a human resolves an escalation they record what the agent should have done. Resolved
escalations can then be exported as new eval tasks, so every hand-off can become a test.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from relay.domain import EscalationReason, Ticket, utcnow


class EscalationStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class Escalation(BaseModel):
    escalation_id: str = Field(default_factory=lambda: f"E-{uuid.uuid4().hex[:10]}")
    ticket_id: str
    run_id: str | None = None
    reason: EscalationReason
    summary: str
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    status: EscalationStatus = EscalationStatus.OPEN
    created_at: datetime = Field(default_factory=utcnow)
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    resolution: str | None = None
    agent_should: Literal["resolve", "escalate"] | None = None
    expected_reason: EscalationReason | None = None


class Resolution(BaseModel):
    resolution: str = Field(min_length=1, description="What the human did or replied.")
    resolved_by: str
    agent_should: Literal["resolve", "escalate"] = Field(
        description="With hindsight: could the agent have resolved this, or was escalating right?"
    )
    expected_reason: EscalationReason | None = None


class EscalationStore(Protocol):
    def add(self, escalation: Escalation) -> Escalation: ...

    def get(self, escalation_id: str) -> Escalation | None: ...

    def list(self, status: EscalationStatus | None = None) -> list[Escalation]: ...

    def save(self, escalation: Escalation) -> Escalation: ...


class InMemoryEscalationStore:
    def __init__(self) -> None:
        self._items: dict[str, Escalation] = {}

    def add(self, escalation: Escalation) -> Escalation:
        self._items[escalation.escalation_id] = escalation
        return escalation

    def get(self, escalation_id: str) -> Escalation | None:
        return self._items.get(escalation_id)

    def list(self, status: EscalationStatus | None = None) -> list[Escalation]:
        items = sorted(self._items.values(), key=lambda e: e.created_at)
        return [e for e in items if status is None or e.status == status]

    def save(self, escalation: Escalation) -> Escalation:
        self._items[escalation.escalation_id] = escalation
        return escalation


class EscalationService:
    """Implements the tools' EscalationSink and the human side of the hand-off."""

    def __init__(
        self,
        store: EscalationStore,
        on_escalated: Callable[[Escalation], None] | None = None,
    ) -> None:
        self.store = store
        self.on_escalated = on_escalated

    def escalate(
        self,
        ticket_id: str,
        reason: str,
        summary: str,
        run_id: str | None,
        trajectory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        escalation = self.store.add(
            Escalation(
                ticket_id=ticket_id,
                run_id=run_id,
                reason=EscalationReason(reason),
                summary=summary,
                trajectory=_compact(trajectory),
            )
        )
        if self.on_escalated:
            self.on_escalated(escalation)
        return escalation.model_dump(mode="json")

    def resolve(self, escalation_id: str, resolution: Resolution) -> Escalation:
        escalation = self.store.get(escalation_id)
        if escalation is None:
            raise KeyError(escalation_id)
        updated = escalation.model_copy(
            update={
                "status": EscalationStatus.RESOLVED,
                "resolved_at": utcnow(),
                "resolved_by": resolution.resolved_by,
                "resolution": resolution.resolution,
                "agent_should": resolution.agent_should,
                "expected_reason": resolution.expected_reason,
            }
        )
        return self.store.save(updated)


def _compact(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the trajectory readable for a human: drop bulky results, keep errors and flags."""
    out = []
    for step in trajectory:
        item = {k: v for k, v in step.items() if k != "result"}
        result = step.get("result")
        if isinstance(result, dict) and "error" in result:
            item["result"] = result
        out.append(item)
    return out


def to_eval_task(escalation: Escalation, ticket: Ticket) -> dict[str, Any]:
    """Turn a resolved escalation into a task for the trajectory eval."""
    if escalation.status != EscalationStatus.RESOLVED or escalation.agent_should is None:
        raise ValueError(f"{escalation.escalation_id} is not resolved yet")
    if escalation.agent_should == "resolve":
        expect: dict[str, Any] = {
            "outcome": "resolved",
            "must_call": ["lookup_ticket", "draft_reply"],
        }
    else:
        reason = escalation.expected_reason or escalation.reason
        expect = {
            "outcome": "escalated",
            "reason": reason.value,
            "must_call": ["lookup_ticket", "escalate_to_human"],
            "must_not_call": ["draft_reply"],
        }
    return {
        "id": f"esc-{escalation.escalation_id}",
        "category": "from_escalation",
        "source": {"escalation_id": escalation.escalation_id, "run_id": escalation.run_id},
        # Keep the ticket's age at hand-off rather than a wall-clock time, so the task replays the
        # same SLA situation under the eval's frozen clock.
        "ticket": ticket.for_model()
        | {
            "created_minutes_ago": max(
                0, int((escalation.created_at - ticket.created_at).total_seconds() // 60)
            )
        },
        "expect": expect,
        "reference_reply": escalation.resolution,
        "notes": f"Agent escalated with {escalation.reason.value}; human said agent should "
        f"{escalation.agent_should}.",
    }
