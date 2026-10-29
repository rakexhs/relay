"""Build the in-memory world for one task. Time is frozen so runs are reproducible."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from evals.schema import Task
from relay.domain import Customer, Ticket
from relay.tools.backends import InMemoryLogs, load_fixture_docs
from relay.tools.world import InMemoryWorld

EVAL_NOW = datetime(2026, 3, 2, 15, 0, tzinfo=UTC)


def build_world(task: Task, now: datetime = EVAL_NOW) -> tuple[InMemoryWorld, Ticket]:
    world = InMemoryWorld(now=now, docs=[*load_fixture_docs(), *task.fixtures.docs])
    spec = task.ticket
    created_at = (
        datetime.fromisoformat(spec.created_at)
        if spec.created_at
        else now - timedelta(minutes=spec.created_minutes_ago)
    )
    ticket = Ticket(
        ticket_id=spec.ticket_id,
        customer_id=spec.customer_id,
        subject=spec.subject,
        body=spec.body,
        priority=spec.priority,
        severity=spec.severity,
        channel=spec.channel,
        created_at=created_at,
    )
    customer_spec = task.fixtures.customer
    customer = Customer(
        customer_id=spec.customer_id,
        name=customer_spec.name if customer_spec else "Alex Morgan",
        plan=customer_spec.plan if customer_spec else "standard",
    )
    world.add_ticket(ticket, customer)

    logs = []
    for row in task.fixtures.logs:
        row = dict(row)
        minutes = row.pop("minutes_ago", 30)
        row.setdefault("customer_id", spec.customer_id)
        row["timestamp"] = (now - timedelta(minutes=minutes)).isoformat()
        logs.append(row)
    world.logs = InMemoryLogs(logs, down=task.fixtures.logs_down)

    for memory in task.fixtures.history:
        world.remember(
            spec.customer_id,
            memory.ticket_id,
            memory.summary,
            resolved_at=now - timedelta(days=memory.days_ago),
        )
    return world, ticket
