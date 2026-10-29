"""Row <-> model mapping for the Postgres tables in migrations/001_init.sql."""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from relay.approvals import Approval
from relay.domain import Customer, Ticket
from relay.escalation import Escalation
from relay.sla import SlaPolicy

TICKET_COLUMNS = (
    "ticket_id, customer_id, subject, body, priority, severity, channel, status, "
    "created_at, messages"
)
RUN_COLUMNS = (
    "run_id, ticket_id, status, outcome, pending_approval_id, error, created_at, updated_at"
)
APPROVAL_COLUMNS = (
    "approval_id, key, run_id, ticket_id, tool, args, status, requested_at, decided_at, "
    "reviewer, note, edited_args"
)
ESCALATION_COLUMNS = (
    "escalation_id, ticket_id, run_id, reason, summary, trajectory, status, created_at, "
    "resolved_at, resolved_by, resolution, agent_should, expected_reason"
)


def ticket_from_row(row: dict[str, Any]) -> Ticket:
    return Ticket.model_validate(row)


def ticket_params(ticket: Ticket) -> dict[str, Any]:
    data = ticket.model_dump()
    data["messages"] = Jsonb(data["messages"])
    for key in ("priority", "severity", "channel", "status"):
        data[key] = str(data[key])
    return data


def customer_from_row(row: dict[str, Any]) -> Customer:
    return Customer.model_validate(row)


def approval_from_row(row: dict[str, Any]) -> Approval:
    return Approval.model_validate(row)


def approval_params(approval: Approval) -> dict[str, Any]:
    data = approval.model_dump()
    data["status"] = str(data["status"])
    data["args"] = Jsonb(data["args"])
    data["edited_args"] = Jsonb(data["edited_args"]) if data["edited_args"] is not None else None
    return data


def escalation_from_row(row: dict[str, Any]) -> Escalation:
    return Escalation.model_validate(row)


def escalation_params(escalation: Escalation) -> dict[str, Any]:
    data = escalation.model_dump(mode="json")
    data["trajectory"] = Jsonb(data["trajectory"])
    for key in ("created_at", "resolved_at"):
        data[key] = getattr(escalation, key)
    return data


def sla_from_row(row: dict[str, Any]) -> SlaPolicy:
    return SlaPolicy.model_validate(row)
