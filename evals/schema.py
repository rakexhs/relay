"""The eval task format (one JSON object per line in a tasks file)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from relay.domain import EscalationReason


class TicketSpec(BaseModel):
    ticket_id: str
    customer_id: str
    subject: str
    body: str
    priority: str = "normal"
    severity: str = "minor"
    channel: str = "email"
    created_minutes_ago: int = 20
    created_at: str | None = None  # absolute time, used by tasks exported from escalations


class CustomerSpec(BaseModel):
    name: str
    plan: str = "standard"


class MemorySpec(BaseModel):
    ticket_id: str
    summary: str
    days_ago: int = 14


class Fixtures(BaseModel):
    """The world the agent sees for this task, on top of the shared help center."""

    customer: CustomerSpec | None = None
    docs: list[dict[str, Any]] = Field(default_factory=list)
    logs: list[dict[str, Any]] = Field(default_factory=list)  # each with minutes_ago
    logs_down: bool = False
    history: list[MemorySpec] = Field(default_factory=list)


class Expectation(BaseModel):
    outcome: Literal["resolved", "escalated"]
    reason: EscalationReason | None = None
    must_call: list[str] = Field(default_factory=list)
    must_not_call: list[str] = Field(default_factory=list)
    # Pairs [a, b]: the first call to a must come before the first call to b.
    order: list[tuple[str, str]] = Field(default_factory=list)
    max_tool_calls: int | None = None
    reply_must_mention: list[str] = Field(default_factory=list)


class ForbiddenAction(BaseModel):
    """A write the agent must never make, matched on tool name and argument substrings."""

    tool: str
    args_contains: dict[str, str] = Field(default_factory=dict)
    why: str = ""


class Injection(BaseModel):
    location: Literal["ticket", "docs", "logs", "memory"]
    canary: str = Field(description="Text from the planted instruction that must never be echoed.")
    goal: str = ""


class Task(BaseModel):
    id: str
    category: str
    ticket: TicketSpec
    fixtures: Fixtures = Field(default_factory=Fixtures)
    expect: Expectation
    forbidden: list[ForbiddenAction] = Field(default_factory=list)
    injection: Injection | None = None
    reference_reply: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    source: dict[str, Any] | None = None


def load_tasks(path: str | Path) -> list[Task]:
    tasks = []
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            try:
                tasks.append(Task.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    ids = [t.id for t in tasks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate task ids: {sorted(duplicates)}")
    return tasks
