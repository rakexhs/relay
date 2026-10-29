"""Core domain types shared by tools, the agent, storage and the API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class Priority(StrEnum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class Severity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class TicketStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class Channel(StrEnum):
    EMAIL = "email"
    CHAT = "chat"
    WEB = "web"
    VOICEMAIL = "voicemail"


class EscalationReason(StrEnum):
    SLA_BREACH = "sla_breach"
    LOW_CONFIDENCE = "low_confidence"
    NEEDS_REFUND = "needs_refund"
    TOOL_FAILURE = "tool_failure"
    CUSTOMER_REQUEST = "customer_request"
    GAVE_UP = "gave_up"


class Customer(BaseModel):
    customer_id: str
    name: str
    email: str = ""
    plan: str = "standard"


class TicketCreate(BaseModel):
    customer_id: str = Field(pattern=r"^C-[A-Za-z0-9]+$")
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=20_000)
    priority: Priority = Priority.NORMAL
    severity: Severity = Severity.MINOR
    channel: Channel = Channel.EMAIL


class Ticket(TicketCreate):
    ticket_id: str
    status: TicketStatus = TicketStatus.OPEN
    created_at: datetime = Field(default_factory=utcnow)
    messages: list[dict[str, Any]] = Field(default_factory=list)

    def for_model(self) -> dict[str, Any]:
        """The fields the agent sees about a ticket."""
        return {
            "ticket_id": self.ticket_id,
            "customer_id": self.customer_id,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority.value,
            "severity": self.severity.value,
            "channel": self.channel.value,
        }
