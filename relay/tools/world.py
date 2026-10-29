"""An in-memory world (docs, tickets, logs, memory) that the tools can run against.

The eval harness builds one per task from the task's fixtures; tests and the demo API use it
too. Nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from relay.domain import Customer, Ticket, utcnow
from relay.memory import HashingEmbedder, InMemoryMemoryStore, MemoryRecord
from relay.tools.backends import (
    DocsIndex,
    InMemoryEscalations,
    InMemoryLogs,
    InMemoryTickets,
    Outbox,
    load_fixture_docs,
)
from relay.tools.registry import ToolContext


@dataclass
class InMemoryWorld:
    docs: list[dict[str, Any]] = field(default_factory=load_fixture_docs)
    tickets: InMemoryTickets = field(default_factory=InMemoryTickets)
    logs: InMemoryLogs = field(default_factory=InMemoryLogs)
    memory: InMemoryMemoryStore = field(default_factory=InMemoryMemoryStore)
    outbox: Outbox = field(default_factory=Outbox)
    escalations: InMemoryEscalations = field(default_factory=InMemoryEscalations)
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)
    now: datetime | None = None

    def clock(self) -> datetime:
        return self.now or utcnow()

    def add_ticket(self, ticket: Ticket, customer: Customer | None = None) -> None:
        self.tickets.put_ticket(ticket)
        if customer is not None:
            self.tickets.put_customer(customer)

    def remember(self, customer_id: str, ticket_id: str, summary: str, **extra: Any) -> None:
        self.memory.add(
            MemoryRecord(
                customer_id=customer_id,
                ticket_id=ticket_id,
                summary=summary,
                embedding=self.embedder.embed(summary),
                **extra,
            )
        )

    def context(self) -> ToolContext:
        return ToolContext(
            docs=DocsIndex(self.docs),
            tickets=self.tickets,
            logs=self.logs,
            outbox=self.outbox,
            escalations=self.escalations,
            memory=self.memory,
            embedder=self.embedder,
            clock=self.clock,
        )
