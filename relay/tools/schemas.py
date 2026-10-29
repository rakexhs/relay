"""Argument schemas for the six tools. Every model-issued call is validated against these."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from relay.domain import EscalationReason

TICKET_ID = r"^T-[A-Za-z0-9]+$"
CUSTOMER_ID = r"^C-[A-Za-z0-9]+$"
DURATION_RE = re.compile(r"^(\d{1,4})([mhd])$")


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchDocsArgs(ToolArgs):
    """Search the product documentation and known-issue articles."""

    query: str = Field(min_length=2, max_length=300, description="Plain-language search terms.")
    top_k: int = Field(default=3, ge=1, le=10, description="How many articles to return.")


class LookupTicketArgs(ToolArgs):
    """Fetch a ticket with its message history and the customer's account details."""

    ticket_id: str = Field(pattern=TICKET_ID, description="Ticket id, e.g. T-1042.")


class QueryLogsArgs(ToolArgs):
    """Search application logs.

    `query` is a list of field:value clauses joined by AND, for example
    `service:webhooks AND status_code:502 AND customer_id:C-17`. Prefix a clause with NOT to
    exclude it. Fields: service, level, status_code, customer_id, request_id, message.
    """

    query: str = Field(
        min_length=1, max_length=500, description="field:value clauses joined by AND."
    )
    since: str = Field(default="24h", description="Look-back window such as 15m, 6h or 7d.")
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("since")
    @classmethod
    def _duration(cls, value: str) -> str:
        match = DURATION_RE.match(value)
        if not match:
            raise ValueError(
                "must be a duration like 15m, 6h or 7d (relative words are not accepted)"
            )
        amount, unit = int(match.group(1)), match.group(2)
        minutes = amount * {"m": 1, "h": 60, "d": 1440}[unit]
        if minutes == 0 or minutes > 30 * 1440:
            raise ValueError("must be between 1m and 30d")
        return value


class DraftReplyArgs(ToolArgs):
    """Write a reply to the customer. With send=true it is sent after human approval."""

    ticket_id: str = Field(pattern=TICKET_ID)
    body: str = Field(min_length=20, max_length=4000, description="The full reply text.")
    send: bool = Field(default=False, description="Send to the customer (needs approval).")


class EscalateArgs(ToolArgs):
    """Hand the ticket to a human with a reason code and a short summary of what was tried."""

    ticket_id: str = Field(pattern=TICKET_ID)
    reason: EscalationReason
    summary: str = Field(min_length=10, max_length=2000)


class RecallHistoryArgs(ToolArgs):
    """Recall summaries of this customer's past resolved tickets."""

    customer_id: str = Field(pattern=CUSTOMER_ID)
    query: str | None = Field(default=None, max_length=300, description="Optional focus.")
    top_k: int = Field(default=3, ge=1, le=10)
