"""Pending approvals for write tools (sending a reply, escalating to a human).

The agent writes a pending approval and then pauses on a LangGraph interrupt. A reviewer
approves or rejects it through the API, which resumes the graph with the decision.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from relay.domain import utcnow


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Approval(BaseModel):
    approval_id: str = Field(default_factory=lambda: f"A-{uuid.uuid4().hex[:10]}")
    key: str  # run_id:tool_call_id, makes creation idempotent across resumes
    run_id: str
    ticket_id: str
    tool: str
    args: dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = Field(default_factory=utcnow)
    decided_at: datetime | None = None
    reviewer: str | None = None
    note: str | None = None
    edited_args: dict[str, Any] | None = None


class Decision(BaseModel):
    approved: bool
    reviewer: str = "reviewer"
    note: str | None = None
    edited_args: dict[str, Any] | None = None


class ApprovalStore(Protocol):
    def create(self, approval: Approval) -> Approval: ...

    def get(self, approval_id: str) -> Approval | None: ...

    def list(self, status: ApprovalStatus | None = None) -> list[Approval]: ...

    def decide(self, approval_id: str, decision: Decision) -> Approval: ...


class ApprovalError(RuntimeError):
    pass


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, Approval] = {}

    def create(self, approval: Approval) -> Approval:
        for existing in self._items.values():
            if existing.key == approval.key:
                return existing
        self._items[approval.approval_id] = approval
        return approval

    def get(self, approval_id: str) -> Approval | None:
        return self._items.get(approval_id)

    def list(self, status: ApprovalStatus | None = None) -> list[Approval]:
        items = sorted(self._items.values(), key=lambda a: a.requested_at)
        return [a for a in items if status is None or a.status == status]

    def decide(self, approval_id: str, decision: Decision) -> Approval:
        approval = self._items.get(approval_id)
        if approval is None:
            raise ApprovalError(f"unknown approval {approval_id}")
        if approval.status != ApprovalStatus.PENDING:
            raise ApprovalError(f"approval {approval_id} is already {approval.status}")
        approval = apply_decision(approval, decision)
        self._items[approval_id] = approval
        return approval


def apply_decision(approval: Approval, decision: Decision) -> Approval:
    return approval.model_copy(
        update={
            "status": ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED,
            "decided_at": utcnow(),
            "reviewer": decision.reviewer,
            "note": decision.note,
            "edited_args": decision.edited_args,
        }
    )
