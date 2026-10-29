"""The six tools, their handlers, and the registry that validates and runs every call.

`ToolRegistry.invoke` never raises. Whatever goes wrong (unknown tool, bad arguments, a
malformed log query, a backend outage, a bug) comes back as a structured error payload so
the agent loop can show it to the model and keep going.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from relay import tracing
from relay.domain import utcnow
from relay.memory import Embedder, MemoryStore
from relay.providers.base import ToolSpec
from relay.tools.backends import DocsIndex, EscalationSink, LogBackend, Outbox, TicketStore
from relay.tools.errors import ToolError, ToolErrorCode, error_payload, from_validation_error
from relay.tools.logquery import parse_query, parse_since
from relay.tools.schemas import (
    DraftReplyArgs,
    EscalateArgs,
    LookupTicketArgs,
    QueryLogsArgs,
    RecallHistoryArgs,
    SearchDocsArgs,
    ToolArgs,
)

TOOL_NAMES = (
    "search_docs",
    "lookup_ticket",
    "query_logs",
    "draft_reply",
    "escalate_to_human",
    "recall_customer_history",
)


@dataclass
class ToolContext:
    docs: DocsIndex
    tickets: TicketStore
    logs: LogBackend
    outbox: Outbox
    escalations: EscalationSink
    memory: MemoryStore
    embedder: Embedder
    clock: Callable[[], datetime] = utcnow


@dataclass
class CallMeta:
    run_id: str | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    approved: bool = False


Handler = Callable[[Any, ToolContext, CallMeta], Any | Awaitable[Any]]


@dataclass(frozen=True)
class ToolDef:
    name: str
    args_model: type[ToolArgs]
    handler: Handler
    writes: bool = False

    @property
    def description(self) -> str:
        return inspect.cleandoc(self.args_model.__doc__ or self.name)

    def spec(self) -> ToolSpec:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


# Handlers ----------------------------------------------------------------------------------


def search_docs(args: SearchDocsArgs, ctx: ToolContext, meta: CallMeta) -> dict[str, Any]:
    return {"query": args.query, "results": ctx.docs.search(args.query, args.top_k)}


def lookup_ticket(args: LookupTicketArgs, ctx: ToolContext, meta: CallMeta) -> dict[str, Any]:
    ticket = ctx.tickets.get_ticket(args.ticket_id)
    if ticket is None:
        raise ToolError(
            ToolErrorCode.NOT_FOUND,
            f"No ticket {args.ticket_id}. Use the ticket_id from the ticket you were given.",
            field="ticket_id",
        )
    customer = ctx.tickets.get_customer(ticket.customer_id)
    return {
        "ticket": {
            **ticket.for_model(),
            "status": ticket.status.value,
            "created_at": ticket.created_at.isoformat(),
            "messages": ticket.messages,
        },
        "customer": customer.model_dump() if customer else None,
    }


async def query_logs(args: QueryLogsArgs, ctx: ToolContext, meta: CallMeta) -> dict[str, Any]:
    query = parse_query(args.query)
    hits = await asyncio.to_thread(
        ctx.logs.search, query, parse_since(args.since), args.limit, ctx.clock()
    )
    return {"query": args.query, "since": args.since, "count": len(hits), "hits": hits}


def draft_reply(args: DraftReplyArgs, ctx: ToolContext, meta: CallMeta) -> dict[str, Any]:
    if ctx.tickets.get_ticket(args.ticket_id) is None:
        raise ToolError(ToolErrorCode.NOT_FOUND, f"No ticket {args.ticket_id}.", field="ticket_id")
    if args.send:
        sent = ctx.outbox.send(args.ticket_id, args.body)
        return {"status": "sent", "message_id": sent["message_id"]}
    draft = ctx.outbox.draft(args.ticket_id, args.body)
    return {"status": "drafted", "draft_id": draft["draft_id"]}


def escalate_to_human(args: EscalateArgs, ctx: ToolContext, meta: CallMeta) -> dict[str, Any]:
    record = ctx.escalations.escalate(
        args.ticket_id, args.reason.value, args.summary, meta.run_id, meta.trajectory
    )
    return {
        "status": "escalated",
        "escalation_id": record["escalation_id"],
        "reason": args.reason.value,
    }


def recall_customer_history(
    args: RecallHistoryArgs, ctx: ToolContext, meta: CallMeta
) -> dict[str, Any]:
    embedding = ctx.embedder.embed(args.query) if args.query else None
    hits = ctx.memory.search(args.customer_id, embedding, args.top_k)
    return {
        "customer_id": args.customer_id,
        "memories": [hit.model_dump(mode="json") for hit in hits],
    }


TOOLS: tuple[ToolDef, ...] = (
    ToolDef("search_docs", SearchDocsArgs, search_docs),
    ToolDef("lookup_ticket", LookupTicketArgs, lookup_ticket),
    ToolDef("query_logs", QueryLogsArgs, query_logs),
    ToolDef("draft_reply", DraftReplyArgs, draft_reply, writes=True),
    ToolDef("escalate_to_human", EscalateArgs, escalate_to_human, writes=True),
    ToolDef("recall_customer_history", RecallHistoryArgs, recall_customer_history),
)


# Registry ----------------------------------------------------------------------------------


class ToolRegistry:
    def __init__(self, ctx: ToolContext, approval_tools: Iterable[str] = ()) -> None:
        self.ctx = ctx
        self.tools = {tool.name: tool for tool in TOOLS}
        self.approval_tools = frozenset(approval_tools)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self.tools.values()]

    def validate(self, name: str, raw_args: dict[str, Any]) -> ToolArgs | dict[str, Any]:
        """Return parsed args, or an error payload describing what is wrong with the call."""
        tool = self.tools.get(name)
        if tool is None:
            return error_payload(
                ToolErrorCode.UNKNOWN_TOOL,
                f"There is no tool named '{name}'.",
                field="name",
                allowed=list(self.tools),
            )
        if "__raw__" in raw_args:
            return error_payload(
                ToolErrorCode.INVALID_ARGUMENTS,
                "The arguments were not valid JSON. Send a JSON object.",
            )
        try:
            return tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return from_validation_error(exc)

    def requires_approval(self, name: str, args: ToolArgs) -> bool:
        if name not in self.approval_tools:
            return False
        if isinstance(args, DraftReplyArgs):
            # Saving a draft is harmless; only sending it to the customer is gated.
            return args.send
        return True

    async def invoke(
        self, name: str, raw_args: dict[str, Any], meta: CallMeta | None = None
    ) -> dict[str, Any]:
        meta = meta or CallMeta()
        with tracing.tool_span(name, raw_args, meta.run_id) as span:
            result = await self._invoke(name, raw_args, meta)
            tracing.record_tool_result(span, result, meta.approved)
            return result

    async def _invoke(self, name: str, raw_args: dict[str, Any], meta: CallMeta) -> dict[str, Any]:
        parsed = self.validate(name, raw_args)
        if isinstance(parsed, dict):
            return parsed
        if self.requires_approval(name, parsed) and not meta.approved:
            return error_payload(
                ToolErrorCode.APPROVAL_REQUIRED,
                f"{name} changes customer-visible state and needs human approval first.",
            )
        try:
            result = self.tools[name].handler(parsed, self.ctx, meta)
            if inspect.isawaitable(result):
                result = await result
        except ToolError as exc:
            return exc.payload()
        except Exception as exc:
            return error_payload(
                ToolErrorCode.INTERNAL,
                f"{name} failed unexpectedly ({type(exc).__name__}). Do not retry the same call.",
            )
        return result
