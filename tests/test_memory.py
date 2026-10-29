import asyncio
import os

import pytest

from relay.agent import AgentDeps, LocalToolClient
from relay.agent.runner import AgentRunner
from relay.config import Settings
from relay.domain import Ticket
from relay.memory import HashingEmbedder, InMemoryMemoryStore, MemoryRecord, MemoryWriter
from relay.memory.embeddings import cosine
from relay.providers import StubProvider
from relay.tools import ToolRegistry

from .conftest import NOW, make_world


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(64)
    a = embedder.embed("webhook endpoint disabled")
    assert a == embedder.embed("webhook endpoint disabled")
    assert abs(cosine(a, a) - 1.0) < 1e-9
    assert cosine(a, embedder.embed("webhooks endpoint disabled")) > cosine(
        a, embedder.embed("invoice pdf download")
    )


def test_resolved_run_is_remembered_and_recalled_next_time():
    world = make_world()
    world.memory = InMemoryMemoryStore()
    settings = Settings(_env_file=None, approval_tools=[])
    registry = ToolRegistry(world.context(), approval_tools=[])
    writer = MemoryWriter(StubProvider(), world.embedder, world.memory)
    deps = AgentDeps(StubProvider(), LocalToolClient(registry), settings, clock=world.clock)
    runner = AgentRunner(deps, tickets=world.tickets, memory_writer=writer)

    asyncio.run(runner.start(world.tickets.get_ticket("T-100"), run_id="run-m1"))
    hits = world.memory.search("C-7", world.embedder.embed("webhooks"), 3)
    assert hits and hits[0].ticket_id == "T-100"
    assert hits[0].summary.startswith("Asked about: Webhooks failing")

    follow_up = Ticket(
        ticket_id="T-101",
        customer_id="C-7",
        subject="Webhooks failing again",
        body="Same webhook problem as last week, deliveries are failing.",
        created_at=NOW,
    )
    world.add_ticket(follow_up)
    asyncio.run(runner.start(follow_up, run_id="run-m2"))
    snapshot = asyncio.run(runner.snapshot("run-m2"))
    recall = snapshot["trajectory"][0]
    assert recall["tool"] == "recall_customer_history"
    assert recall["result"]["memories"][0]["ticket_id"] == "T-100"


@pytest.mark.pgvector
@pytest.mark.skipif(
    not os.environ.get("RELAY_TEST_PGVECTOR_URL"), reason="set RELAY_TEST_PGVECTOR_URL to run"
)
def test_pgvector_store_ranks_by_cosine():
    from relay.db.migrate import migrate
    from relay.memory.pgvector_store import PgVectorMemoryStore

    url = os.environ["RELAY_TEST_PGVECTOR_URL"]
    migrate(url)
    store = PgVectorMemoryStore(url)
    embedder = HashingEmbedder(256)
    for ticket_id, summary in [("T-1", "webhook endpoint disabled"), ("T-2", "invoice download")]:
        store.add(
            MemoryRecord(
                customer_id="C-pg",
                ticket_id=ticket_id,
                summary=summary,
                embedding=embedder.embed(summary),
            )
        )
    hits = store.search("C-pg", embedder.embed("webhook failures"), 2)
    assert hits[0].ticket_id == "T-1"
