"""Postgres stores and the approval flow on a real database. Skipped without a database."""

import asyncio
import os
import uuid

import pytest

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not os.environ.get("RELAY_TEST_DATABASE_URL"), reason="set RELAY_TEST_DATABASE_URL to run"
    ),
]


@pytest.fixture(scope="module")
def store():
    from relay.db.migrate import migrate
    from relay.db.repo import PostgresStore

    url = os.environ["RELAY_TEST_DATABASE_URL"]
    migrate(url)
    store = PostgresStore(url)
    yield store
    store.close()


def _ids():
    suffix = uuid.uuid4().hex[:8]
    return f"T-{suffix}", f"C-{suffix}"


def test_ticket_roundtrip_and_status(store):
    from relay.domain import Customer, Priority, Ticket, TicketStatus

    ticket_id, customer_id = _ids()
    store.put_customer(Customer(customer_id=customer_id, name="Ana"))
    store.put_ticket(
        Ticket(
            ticket_id=ticket_id,
            customer_id=customer_id,
            subject="s",
            body="b",
            priority=Priority.HIGH,
        )
    )
    store.set_status(ticket_id, TicketStatus.ESCALATED)
    ticket = store.get_ticket(ticket_id)
    assert ticket.priority == Priority.HIGH
    assert ticket.status == TicketStatus.ESCALATED
    assert store.get_customer(customer_id).name == "Ana"


def test_approval_create_is_idempotent_and_decide_once(store):
    from relay.approvals import Approval, ApprovalError, Decision

    key = f"run-{uuid.uuid4().hex[:6]}:call_1"
    first = store.create(
        Approval(key=key, run_id="r", ticket_id="T-1", tool="draft_reply", args={"a": 1})
    )
    second = store.create(
        Approval(key=key, run_id="r", ticket_id="T-1", tool="draft_reply", args={"a": 1})
    )
    assert first.approval_id == second.approval_id
    decided = store.decide(first.approval_id, Decision(approved=True, reviewer="kim"))
    assert decided.status == "approved"
    with pytest.raises(ApprovalError):
        store.decide(first.approval_id, Decision(approved=False))


def test_sla_policies_are_seeded(store):
    from relay.domain import Priority

    assert store.policies()[Priority.URGENT].resolution_minutes == 240


def test_runner_with_postgres_stores_and_checkpoints(store):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from relay.agent import AgentDeps, LocalToolClient
    from relay.agent.runner import AgentRunner, RunStatus
    from relay.approvals import Decision
    from relay.config import Settings
    from relay.db.repo import PostgresEscalationStore
    from relay.escalation import EscalationService
    from relay.providers import StubProvider
    from relay.tools import ToolRegistry

    from .conftest import make_world

    world = make_world()
    ticket = world.tickets.get_ticket("T-100").model_copy(update={"ticket_id": _ids()[0]})
    store.put_customer(world.tickets.get_customer("C-7"))
    store.put_ticket(ticket)
    world.tickets = store  # tools now read tickets from Postgres
    world.escalations = EscalationService(PostgresEscalationStore(store))
    settings = Settings(_env_file=None)
    registry = ToolRegistry(world.context(), approval_tools=settings.approval_tools)

    async def go():
        async with AsyncPostgresSaver.from_conn_string(
            os.environ["RELAY_TEST_DATABASE_URL"]
        ) as saver:
            await saver.setup()
            deps = AgentDeps(
                StubProvider(),
                LocalToolClient(registry),
                settings,
                clock=world.clock,
                approvals=store,
            )
            runner = AgentRunner(
                deps, checkpointer=saver, runs=store, tickets=store, sla_policies=store
            )
            run = await runner.start(ticket)
            assert run.status == RunStatus.AWAITING_APPROVAL
            assert store.get_run(run.run_id).pending_approval_id == run.pending_approval_id
            # A fresh runner (as after a restart) can resume from Postgres alone.
            runner2 = AgentRunner(
                deps, checkpointer=saver, runs=store, tickets=store, sla_policies=store
            )
            return await runner2.decide(run.pending_approval_id, Decision(approved=True))

    run = asyncio.run(go())
    assert run.status == RunStatus.COMPLETED
    assert store.get_ticket(ticket.ticket_id).status == "resolved"
