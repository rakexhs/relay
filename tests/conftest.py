from datetime import UTC, datetime, timedelta

import pytest

from relay.domain import Customer, Priority, Ticket
from relay.tools import ToolRegistry
from relay.tools.backends import InMemoryLogs
from relay.tools.world import InMemoryWorld

NOW = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)


def make_world(*, logs_down: bool = False) -> InMemoryWorld:
    world = InMemoryWorld(now=NOW)
    world.add_ticket(
        Ticket(
            ticket_id="T-100",
            customer_id="C-7",
            subject="Webhooks failing with 502 since yesterday",
            body="Our endpoint stopped getting webhook events and the dashboard shows 502 errors.",
            priority=Priority.HIGH,
            created_at=NOW - timedelta(minutes=30),
        ),
        Customer(customer_id="C-7", name="Dana Ortiz", plan="business"),
    )
    world.logs = InMemoryLogs(
        [
            {
                "timestamp": (NOW - timedelta(hours=3)).isoformat(),
                "service": "webhooks",
                "level": "error",
                "status_code": 502,
                "customer_id": "C-7",
                "request_id": "req_a1",
                "message": "delivery failed: upstream returned 502",
            },
            {
                "timestamp": (NOW - timedelta(days=3)).isoformat(),
                "service": "webhooks",
                "level": "error",
                "status_code": 502,
                "customer_id": "C-7",
                "request_id": "req_old",
                "message": "delivery failed: upstream returned 502",
            },
            {
                "timestamp": (NOW - timedelta(hours=1)).isoformat(),
                "service": "api",
                "level": "info",
                "status_code": 200,
                "customer_id": "C-7",
                "request_id": "req_b2",
                "message": "ok",
            },
        ],
        down=logs_down,
    )
    world.remember(
        "C-7", "T-050", "Webhook endpoint disabled after 24h of failures; re-enabled it."
    )
    world.remember("C-7", "T-051", "Asked how to download invoices as PDF.")
    return world


@pytest.fixture
def world() -> InMemoryWorld:
    return make_world()


@pytest.fixture
def registry(world) -> ToolRegistry:
    return ToolRegistry(world.context(), approval_tools=["draft_reply", "escalate_to_human"])
