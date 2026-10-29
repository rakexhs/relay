"""SLA timers: resolution deadlines per priority, queue ranking and breach detection."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import BaseModel

from relay.domain import Priority, Severity, Ticket, TicketStatus


class SlaPolicy(BaseModel):
    priority: Priority
    first_response_minutes: int
    resolution_minutes: int


DEFAULT_POLICIES: dict[Priority, SlaPolicy] = {
    p.priority: p
    for p in (
        SlaPolicy(priority=Priority.URGENT, first_response_minutes=30, resolution_minutes=240),
        SlaPolicy(priority=Priority.HIGH, first_response_minutes=120, resolution_minutes=1440),
        SlaPolicy(priority=Priority.NORMAL, first_response_minutes=480, resolution_minutes=2880),
        SlaPolicy(priority=Priority.LOW, first_response_minutes=1440, resolution_minutes=7200),
    )
}

SEVERITY_RANK = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2}
PRIORITY_RANK = {Priority.URGENT: 0, Priority.HIGH: 1, Priority.NORMAL: 2, Priority.LOW: 3}
OPEN_STATES = {TicketStatus.OPEN, TicketStatus.IN_PROGRESS, TicketStatus.AWAITING_APPROVAL}


class SlaPolicyStore(Protocol):
    def policies(self) -> dict[Priority, SlaPolicy]: ...


class StaticSlaPolicies:
    def __init__(self, policies: dict[Priority, SlaPolicy] | None = None) -> None:
        self._policies = dict(policies or DEFAULT_POLICIES)

    def policies(self) -> dict[Priority, SlaPolicy]:
        return self._policies


class SlaStatus(BaseModel):
    ticket_id: str
    priority: Priority
    severity: Severity
    deadline: datetime
    remaining_minutes: float
    breached: bool


def deadline(ticket: Ticket, policies: dict[Priority, SlaPolicy]) -> datetime:
    policy = policies.get(ticket.priority) or DEFAULT_POLICIES[ticket.priority]
    return ticket.created_at + timedelta(minutes=policy.resolution_minutes)


def status(ticket: Ticket, policies: dict[Priority, SlaPolicy], now: datetime) -> SlaStatus:
    due = deadline(ticket, policies)
    remaining = (due - now).total_seconds() / 60
    return SlaStatus(
        ticket_id=ticket.ticket_id,
        priority=ticket.priority,
        severity=ticket.severity,
        deadline=due,
        remaining_minutes=round(remaining, 1),
        breached=remaining <= 0,
    )


def rank(
    tickets: Iterable[Ticket], policies: dict[Priority, SlaPolicy], now: datetime
) -> list[SlaStatus]:
    """Open tickets, most urgent first: breached, then least time left, then severity."""
    statuses = [status(t, policies, now) for t in tickets if t.status in OPEN_STATES]
    return sorted(
        statuses,
        key=lambda s: (
            not s.breached,
            s.remaining_minutes,
            SEVERITY_RANK[s.severity],
            PRIORITY_RANK[s.priority],
        ),
    )


def breached(
    tickets: Iterable[Ticket], policies: dict[Priority, SlaPolicy], now: datetime
) -> list[SlaStatus]:
    return [s for s in rank(tickets, policies, now) if s.breached]
