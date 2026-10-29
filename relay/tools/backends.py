"""Backends the tools read from and write to.

In-memory implementations back tests, the eval harness and the demo API. The OpenSearch log
backend and the Postgres stores (relay.db) plug in behind the same small interfaces.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from relay.domain import Customer, Ticket, TicketStatus, utcnow
from relay.tools.errors import ToolError, ToolErrorCode
from relay.tools.logquery import LogQuery, to_opensearch

WORD_RE = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an and are as at be but by for from has have how i if in is it my no not of on or our "
    "so that the this to was we what when why with you your".split()
)


def _terms(text: str) -> list[str]:
    out = []
    for token in WORD_RE.findall(text.lower()):
        if token in STOP:
            continue
        out.append(token[:-1] if len(token) > 4 and token.endswith("s") else token)
    return out


# Docs --------------------------------------------------------------------------------------


class DocsIndex:
    """Keyword search over help articles, scored as weighted query-term coverage in [0, 1]."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self._terms = [Counter(_terms(f"{d['title']} {d['title']} {d['body']}")) for d in docs]
        df: Counter[str] = Counter()
        for counts in self._terms:
            df.update(counts.keys())
        n = max(len(docs), 1)
        self._idf = {term: math.log(1 + n / count) for term, count in df.items()}

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        q_terms = list(dict.fromkeys(_terms(query)))
        if not q_terms:
            return []
        default_idf = math.log(1 + max(len(self.docs), 1))
        weights = {t: self._idf.get(t, default_idf) for t in q_terms}
        total = sum(weights.values())
        scored = []
        for doc, counts in zip(self.docs, self._terms, strict=True):
            covered = sum(w for t, w in weights.items() if counts.get(t))
            if covered == 0:
                continue
            tf_bonus = sum(min(counts.get(t, 0), 3) for t in q_terms) / (3 * len(q_terms))
            score = 0.85 * covered / total + 0.15 * tf_bonus
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "snippet": doc.get("answer") or doc["body"][:400],
                "score": round(score, 3),
            }
            for score, doc in scored[:top_k]
        ]


# Tickets -----------------------------------------------------------------------------------


class TicketStore(Protocol):
    def get_ticket(self, ticket_id: str) -> Ticket | None: ...

    def get_customer(self, customer_id: str) -> Customer | None: ...


class InMemoryTickets:
    def __init__(
        self, tickets: list[Ticket] | None = None, customers: list[Customer] | None = None
    ) -> None:
        self.tickets = {t.ticket_id: t for t in tickets or []}
        self.customers = {c.customer_id: c for c in customers or []}

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        return self.tickets.get(ticket_id)

    def get_customer(self, customer_id: str) -> Customer | None:
        return self.customers.get(customer_id)

    def put_ticket(self, ticket: Ticket) -> None:
        self.tickets[ticket.ticket_id] = ticket

    def put_customer(self, customer: Customer) -> None:
        self.customers[customer.customer_id] = customer

    def set_status(self, ticket_id: str, status: TicketStatus) -> None:
        ticket = self.tickets.get(ticket_id)
        if ticket is not None:
            self.tickets[ticket_id] = ticket.model_copy(update={"status": status})

    def list_tickets(self) -> list[Ticket]:
        return list(self.tickets.values())


# Logs --------------------------------------------------------------------------------------


class LogBackend(Protocol):
    def search(
        self, query: LogQuery, since: timedelta, limit: int, now: datetime
    ) -> list[dict[str, Any]]: ...


class InMemoryLogs:
    def __init__(self, records: list[dict[str, Any]] | None = None, *, down: bool = False) -> None:
        self.records = [dict(r) for r in records or []]
        self.down = down
        for record in self.records:
            ts = record.get("timestamp")
            if isinstance(ts, str):
                record["timestamp"] = datetime.fromisoformat(ts)

    def search(
        self, query: LogQuery, since: timedelta, limit: int, now: datetime
    ) -> list[dict[str, Any]]:
        if self.down:
            raise ToolError(
                ToolErrorCode.BACKEND_UNAVAILABLE,
                "The log cluster did not respond. Try again shortly or continue without logs.",
                retryable=True,
            )
        start = now - since
        hits = [
            r
            for r in self.records
            if r["timestamp"] >= start and r["timestamp"] <= now and query.matches(r)
        ]
        hits.sort(key=lambda r: r["timestamp"], reverse=True)
        return [{**r, "timestamp": r["timestamp"].isoformat()} for r in hits[:limit]]


class OpenSearchLogs:
    """Log search against an OpenSearch index. Needs the `opensearch` extra."""

    def __init__(self, url: str, index: str, client: Any | None = None) -> None:
        if client is None:
            try:
                from opensearchpy import OpenSearch
            except ImportError as exc:
                from relay.providers.base import MissingDependencyError

                raise MissingDependencyError("opensearch-py", "opensearch") from exc
            client = OpenSearch(hosts=[url], timeout=10)
        self._client = client
        self.index = index

    def search(
        self, query: LogQuery, since: timedelta, limit: int, now: datetime
    ) -> list[dict[str, Any]]:
        minutes = int(since.total_seconds() // 60)
        body = to_opensearch(query, f"{minutes}m", limit)
        try:
            response = self._client.search(index=self.index, body=body)
        except Exception as exc:  # the client raises several transport error types
            raise ToolError(
                ToolErrorCode.BACKEND_UNAVAILABLE,
                f"OpenSearch query failed: {type(exc).__name__}",
                retryable=True,
            ) from exc
        hits = []
        for hit in response.get("hits", {}).get("hits", []):
            source = dict(hit.get("_source", {}))
            source["timestamp"] = source.pop("@timestamp", source.get("timestamp"))
            hits.append(source)
        return hits


# Outbound writes ---------------------------------------------------------------------------


class Outbox:
    """Where replies go. Sent messages are what the eval checks for forbidden content."""

    def __init__(self) -> None:
        self.drafts: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []

    def draft(self, ticket_id: str, body: str) -> dict[str, Any]:
        record = {"draft_id": f"D-{uuid.uuid4().hex[:8]}", "ticket_id": ticket_id, "body": body}
        self.drafts.append(record)
        return record

    def send(self, ticket_id: str, body: str) -> dict[str, Any]:
        record = {
            "message_id": f"M-{uuid.uuid4().hex[:8]}",
            "ticket_id": ticket_id,
            "body": body,
            "sent_at": utcnow().isoformat(),
        }
        self.sent.append(record)
        return record


class EscalationSink(Protocol):
    def escalate(
        self,
        ticket_id: str,
        reason: str,
        summary: str,
        run_id: str | None,
        trajectory: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class InMemoryEscalations:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def escalate(
        self,
        ticket_id: str,
        reason: str,
        summary: str,
        run_id: str | None,
        trajectory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        record = {
            "escalation_id": f"E-{uuid.uuid4().hex[:8]}",
            "ticket_id": ticket_id,
            "reason": reason,
            "summary": summary,
            "run_id": run_id,
            "trajectory": trajectory,
            "status": "open",
            "created_at": utcnow().isoformat(),
        }
        self.items.append(record)
        return record


# Fixtures ----------------------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_fixture_docs() -> list[dict[str, Any]]:
    return read_jsonl(FIXTURES / "docs.jsonl")
