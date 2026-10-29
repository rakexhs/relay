"""A deterministic, offline provider used by tests, CI and the default eval run.

The stub is a rule-based support policy, not a language model. It reads the same messages a
real model would see (ticket, tool results, structured errors) and decides the next tool call
with fixed rules. That makes the whole agent path, the approval gate and the eval scorers
runnable without API keys, and makes eval reports reproducible run to run.

Some of its habits are deliberately imperfect so the harness has something to measure: its
first log query uses `status:` instead of `status_code:` and passes `since="yesterday"`
verbatim, the way a hurried model often does. It only recovers by reading the structured
error that `query_logs` returns.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from relay.providers.base import Message, ModelProvider, ModelRequest, ModelResponse, ToolCall

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have hi hello
    how i i'm im if in into is it it's its just me my no not of on or our please so some that
    the their them then there this to too up us was we were what when where which who why will
    with you your yours thanks thank hey since still also any all get got getting can't cannot
    dont don't any been being would should about after again""".split()
)

INCIDENT_RE = re.compile(
    r"\b(error|errors|5\d\d|4\d\d|timeout|timed out|crash\w*|failing|failed|fails|outage|down|"
    r"slow|latency|not delivered|not firing)\b",
    re.I,
)
HUMAN_RE = re.compile(
    r"\b(speak|talk|chat)\s+(to|with)\s+(a|an)?\s*(human|person|agent|manager|someone)\b|"
    r"\breal person\b",
    re.I,
)
REFUND_RE = re.compile(r"\b(refund|chargeback|money back|reimburse\w*)\b", re.I)
HTTP_CODE_RE = re.compile(r"\b([45]\d\d)\b")
LAST_N_RE = re.compile(r"\blast\s+(\d+)\s+(minute|hour|day)s?\b", re.I)

SERVICE_HINTS = (
    ("webhooks", re.compile(r"webhook", re.I)),
    ("auth", re.compile(r"\b(login|log in|sign in|sso|password|2fa|mfa)\b", re.I)),
    ("billing", re.compile(r"\b(invoice|billing|charged|card|payment)\b", re.I)),
    ("exports", re.compile(r"\bexport", re.I)),
    ("sync", re.compile(r"\bsync", re.I)),
)

CONFIDENCE_THRESHOLD = 0.3
MAX_FAILURES_PER_TOOL = 3


def iter_json_objects(text: str) -> Iterator[Any]:
    """Yield every top-level JSON object embedded in free text."""
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start == -1:
            return
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        yield obj
        index = end


def first_json(text: str) -> Any:
    return next(iter_json_objects(text), None)


def keywords(text: str, limit: int = 8) -> list[str]:
    seen: list[str] = []
    for token in re.findall(r"[a-z0-9][a-z0-9_\-]+", text.lower()):
        if token in STOPWORDS or token.isdigit() or token in seen:
            continue
        seen.append(token)
    return seen[:limit]


@dataclass
class Observation:
    call: ToolCall
    result: Any

    @property
    def error(self) -> dict[str, Any] | None:
        if isinstance(self.result, dict) and isinstance(self.result.get("error"), dict):
            return self.result["error"]
        return None


@dataclass
class TicketView:
    """What the stub knows so far, rebuilt from the message history on every call."""

    ticket: dict[str, Any]
    observations: list[Observation] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.ticket.get('subject', '')}\n{self.ticket.get('body', '')}"

    def calls(self, name: str) -> list[Observation]:
        return [o for o in self.observations if o.call.name == name]

    def succeeded(self, name: str) -> Observation | None:
        for obs in reversed(self.calls(name)):
            if obs.error is None:
                return obs
        return None

    def failures(self, name: str) -> int:
        return sum(1 for o in self.calls(name) if o.error is not None)

    def stuck(self, name: str) -> bool:
        """True when a tool keeps failing: too many errors, or the same error twice running."""
        if self.succeeded(name):
            return False
        errors = [o.error for o in self.calls(name)]
        if sum(e is not None for e in errors) >= MAX_FAILURES_PER_TOOL:
            return True
        last_two = errors[-2:]
        return (
            len(last_two) == 2
            and all(last_two)
            and (last_two[0].get("code"), last_two[0].get("field"))
            == (last_two[1].get("code"), last_two[1].get("field"))
        )


def read_view(messages: list[Message]) -> TicketView:
    ticket: dict[str, Any] = {}
    memory: list[dict[str, Any]] = []
    pending: dict[str, ToolCall] = {}
    observations: list[Observation] = []
    for msg in messages:
        if msg.role == "user" and not ticket:
            for obj in iter_json_objects(msg.content):
                if isinstance(obj, dict) and "ticket_id" in obj and not ticket:
                    ticket = obj
                elif isinstance(obj, dict) and "memories" in obj:
                    memory = list(obj["memories"])
        elif msg.role == "assistant":
            for call in msg.tool_calls:
                pending[call.id] = call
        elif msg.role == "tool" and msg.tool_call_id in pending:
            observations.append(Observation(pending.pop(msg.tool_call_id), first_json(msg.content)))
    return TicketView(ticket=ticket, observations=observations, memory=memory)


class StubProvider(ModelProvider):
    name = "stub"

    def __init__(self, model: str = "stub-policy-v1") -> None:
        super().__init__(model)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        view = read_view(request.messages)
        if request.purpose == "plan":
            return ModelResponse(text=self.plan(view))
        if request.purpose == "summarize":
            return ModelResponse(text=self.summarize(request.messages))
        if request.purpose == "act":
            return self.act(view)
        raise ValueError(f"stub provider does not handle purpose={request.purpose!r}")

    # Planning -----------------------------------------------------------------

    def plan(self, view: TicketView) -> str:
        steps = ["Look up the ticket for account details and prior messages."]
        if HUMAN_RE.search(view.text):
            steps.append("The customer asked for a person: hand off with customer_request.")
            return _numbered(steps)
        steps.append(f"Search the docs for: {' '.join(keywords(view.text, 5))}.")
        if REFUND_RE.search(view.text):
            steps.append("Refunds need a human: escalate with needs_refund after checking policy.")
            return _numbered(steps)
        if INCIDENT_RE.search(view.text):
            steps.append(f"Query {_guess_service(view.text)} logs for this customer's errors.")
        steps.append("Reply with the documented fix, or escalate with low_confidence if none fits.")
        if any(o.error for o in view.observations):
            steps.insert(1, "Fix the rejected tool call using the error hint before going on.")
        return _numbered(steps)

    # Acting -------------------------------------------------------------------

    def act(self, view: TicketView) -> ModelResponse:
        call = self._next_call(view)
        if call is None:
            return ModelResponse(text=self._final_note(view))
        return ModelResponse(tool_calls=[call])

    def _next_call(self, view: TicketView) -> ToolCall | None:
        ticket_id = view.ticket.get("ticket_id", "")
        text = view.text

        if _finished(view):
            return None

        for tool in ("lookup_ticket", "search_docs", "query_logs"):
            if view.stuck(tool):
                return _escalate(
                    ticket_id, "tool_failure", f"{tool} kept failing: {_last_error(view, tool)}"
                )

        if not view.succeeded("lookup_ticket"):
            return ToolCall(name="lookup_ticket", args={"ticket_id": ticket_id})

        if HUMAN_RE.search(text):
            return _escalate(ticket_id, "customer_request", "Customer asked to talk to a person.")

        if not view.succeeded("search_docs"):
            return ToolCall(
                name="search_docs", args={"query": " ".join(keywords(text)), "top_k": 3}
            )

        if REFUND_RE.search(text):
            return _escalate(
                ticket_id,
                "needs_refund",
                "Customer is asking for a refund; agents cannot issue one.",
            )

        if INCIDENT_RE.search(text) and not view.succeeded("query_logs"):
            failed = [o for o in view.calls("query_logs") if o.error]
            if failed:
                return ToolCall(
                    name="query_logs", args=_repair_log_args(failed[-1].call.args, failed[-1].error)
                )
            return ToolCall(name="query_logs", args=_first_log_args(view))

        rejected = _rejected_draft(view)
        if rejected is not None:
            return _escalate(
                ticket_id, "low_confidence", f"Reviewer rejected the draft: {rejected}"
            )

        docs = (view.succeeded("search_docs").result or {}).get("results", [])
        top = docs[0] if docs else None
        if top is None or float(top.get("score", 0)) < CONFIDENCE_THRESHOLD:
            return _escalate(
                ticket_id, "low_confidence", "No documentation matched this ticket closely enough."
            )
        return ToolCall(
            name="draft_reply",
            args={"ticket_id": ticket_id, "body": _compose_reply(view, top), "send": True},
        )

    def _final_note(self, view: TicketView) -> str:
        if view.succeeded("escalate_to_human"):
            return "Escalated to a human with the trajectory attached."
        return "Reply sent to the customer."

    # Summaries ----------------------------------------------------------------

    def summarize(self, messages: list[Message]) -> str:
        text = "\n".join(m.content for m in messages if m.role == "user")
        payload = first_json(text) or {}
        subject = payload.get("subject", "")
        resolution = payload.get("resolution", "")
        summary = f"Asked about: {subject}."
        if resolution:
            summary += f" Resolved by: {resolution}"
        return summary.strip()


def _numbered(steps: list[str]) -> str:
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def _guess_service(text: str) -> str:
    for service, pattern in SERVICE_HINTS:
        if pattern.search(text):
            return service
    return "api"


def _escalate(ticket_id: str, reason: str, summary: str) -> ToolCall:
    return ToolCall(
        name="escalate_to_human",
        args={"ticket_id": ticket_id, "reason": reason, "summary": summary},
    )


def _finished(view: TicketView) -> bool:
    if view.succeeded("escalate_to_human"):
        return True
    sent = view.succeeded("draft_reply")
    return bool(sent and isinstance(sent.result, dict) and sent.result.get("status") == "sent")


def _rejected_draft(view: TicketView) -> str | None:
    for obs in reversed(view.calls("draft_reply")):
        if isinstance(obs.result, dict) and obs.result.get("status") == "rejected":
            return obs.result.get("note") or "no note"
    return None


def _last_error(view: TicketView, tool: str) -> str:
    for obs in reversed(view.calls(tool)):
        if obs.error:
            return obs.error.get("code", "unknown")
    return "unknown"


def _first_log_args(view: TicketView) -> dict[str, Any]:
    text = view.text
    clauses = [f"service:{_guess_service(text)}"]
    code = HTTP_CODE_RE.search(text)
    # The stub's first draft uses `status:`, which is not a real field. query_logs answers
    # with invalid_query and the list of allowed fields, which is what the repair reads.
    clauses.append(f"status:{code.group(1)}" if code else "level:error")
    customer = view.ticket.get("customer_id")
    if customer:
        clauses.append(f"customer_id:{customer}")
    since = "24h"
    if re.search(r"\byesterday\b", text, re.I):
        since = "yesterday"
    elif match := LAST_N_RE.search(text):
        since = f"{match.group(1)}{match.group(2)[0].lower()}"
    elif re.search(r"\b(this morning|today)\b", text, re.I):
        since = "12h"
    return {"query": " AND ".join(clauses), "since": since, "limit": 20}


def _repair_log_args(args: dict[str, Any], error: dict[str, Any] | None) -> dict[str, Any]:
    fixed = dict(args)
    if not error:
        return fixed
    field_name = error.get("field")
    if field_name == "since":
        fixed["since"] = "48h"
    elif field_name == "query":
        allowed = error.get("allowed") or []
        clauses = []
        for clause in str(args.get("query", "")).split(" AND "):
            name, _, value = clause.partition(":")
            if allowed and name not in allowed:
                match = difflib.get_close_matches(name, allowed, n=1, cutoff=0.4)
                if not match:
                    continue
                name = match[0]
            clauses.append(f"{name}:{value}")
        fixed["query"] = " AND ".join(clauses) or "level:error"
    elif field_name == "limit":
        fixed["limit"] = 20
    return fixed


def _compose_reply(view: TicketView, doc: dict[str, Any]) -> str:
    lookup = view.succeeded("lookup_ticket")
    customer = (lookup.result or {}).get("customer", {}) if lookup else {}
    name = customer.get("name", "").split(" ")[0] or "there"
    lines = [
        f"Hi {name},",
        "",
        f'Thanks for reaching out about "{view.ticket.get("subject", "")}".',
    ]
    lines += ["", doc.get("snippet", "").strip()]
    logs = view.succeeded("query_logs")
    hits = (logs.result or {}).get("hits", []) if logs else []
    if hits:
        latest = hits[0]
        lines += [
            "",
            f"I checked our logs: {len(hits)} matching event(s) from {latest.get('service')}, "
            f'most recently "{latest.get("message")}" (request {latest.get("request_id")}).',
        ]
    if view.memory:
        lines += ["", f"This looks related to an earlier ticket: {view.memory[0].get('summary')}"]
    lines += [
        "",
        f"More detail: {doc.get('title')} ({doc.get('doc_id')}).",
        "",
        "If this doesn't sort it out, reply here and we'll keep digging.",
        "",
        "Relay Support",
    ]
    return "\n".join(lines)
