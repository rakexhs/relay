"""Wire settings into concrete stores, tools, provider and runner for the API process."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from relay.agent import AgentDeps, LocalToolClient, ToolClient
from relay.agent.checkpoint import open_checkpointer
from relay.agent.runner import AgentRunner, InMemoryRunStore
from relay.approvals import ApprovalStore, InMemoryApprovalStore
from relay.config import Settings
from relay.domain import utcnow
from relay.escalation import EscalationService, InMemoryEscalationStore
from relay.memory import HashingEmbedder, InMemoryMemoryStore, MemoryStore, MemoryWriter
from relay.providers import ModelProvider, get_provider
from relay.sla import SlaPolicyStore, StaticSlaPolicies
from relay.tools import ToolContext, ToolRegistry
from relay.tools.backends import (
    DocsIndex,
    InMemoryLogs,
    InMemoryTickets,
    LogBackend,
    OpenSearchLogs,
    Outbox,
    load_demo_customers,
    load_demo_logs,
    load_fixture_docs,
)
from relay.voice import Transcriber, WhisperTranscriber

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    provider: ModelProvider
    tickets: Any  # InMemoryTickets or PostgresStore
    runs: Any
    approvals: ApprovalStore
    escalations: EscalationService
    sla_policies: SlaPolicyStore
    memory_writer: MemoryWriter
    registry: ToolRegistry
    tools: ToolClient
    runner: AgentRunner
    transcriber: Transcriber
    outbox: Outbox = field(default_factory=Outbox)
    storage: str = "memory"


def _memory_store(settings: Settings, pool: Any) -> MemoryStore:
    if pool is None:
        return InMemoryMemoryStore()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'customer_memory'"
        ).fetchone()
    if row is None:
        log.warning("customer_memory table missing (pgvector not installed?); using in-memory")
        return InMemoryMemoryStore()
    from relay.memory.pgvector_store import PgVectorMemoryStore

    return PgVectorMemoryStore(pool)


def _log_backend(settings: Settings) -> LogBackend:
    if settings.opensearch_url:
        return OpenSearchLogs(settings.opensearch_url, settings.log_index)
    return InMemoryLogs(load_demo_logs(utcnow()))


@asynccontextmanager
async def build_services(
    settings: Settings,
    *,
    provider: ModelProvider | None = None,
    transcriber: Transcriber | None = None,
    tool_client: ToolClient | None = None,
) -> AsyncIterator[Services]:
    async with AsyncExitStack() as stack:
        provider = provider or get_provider(settings)
        if settings.database_url:
            from relay.db.migrate import migrate
            from relay.db.repo import PostgresEscalationStore, PostgresStore

            migrate(settings.database_url)
            store = PostgresStore(settings.database_url)
            stack.callback(store.close)
            for customer in load_demo_customers():
                if store.get_customer(customer.customer_id) is None:
                    store.put_customer(customer)
            tickets = runs = approvals = sla_policies = store
            escalation_store: Any = PostgresEscalationStore(store)
            memory = _memory_store(settings, store.pool)
            storage = "postgres"
        else:
            tickets = InMemoryTickets(customers=load_demo_customers())
            runs = InMemoryRunStore()
            approvals = InMemoryApprovalStore()
            sla_policies = StaticSlaPolicies()
            escalation_store = InMemoryEscalationStore()
            memory = InMemoryMemoryStore()
            storage = "memory"

        embedder = HashingEmbedder(settings.embedding_dim)
        escalations = EscalationService(escalation_store)
        outbox = Outbox()
        context = ToolContext(
            docs=DocsIndex(load_fixture_docs()),
            tickets=tickets,
            logs=_log_backend(settings),
            outbox=outbox,
            escalations=escalations,
            memory=memory,
            embedder=embedder,
        )
        registry = ToolRegistry(context, approval_tools=settings.approval_tools)
        tools = tool_client or LocalToolClient(registry)
        checkpointer = await stack.enter_async_context(open_checkpointer(settings))
        memory_writer = MemoryWriter(provider, embedder, memory)
        deps = AgentDeps(provider, tools, settings, approvals=approvals)
        runner = AgentRunner(
            deps,
            checkpointer=checkpointer,
            runs=runs,
            tickets=tickets,
            sla_policies=sla_policies,
            memory_writer=memory_writer,
        )
        yield Services(
            settings=settings,
            provider=provider,
            tickets=tickets,
            runs=runs,
            approvals=approvals,
            escalations=escalations,
            sla_policies=sla_policies,
            memory_writer=memory_writer,
            registry=registry,
            tools=tools,
            runner=runner,
            transcriber=transcriber or WhisperTranscriber(settings.whisper_model),
            outbox=outbox,
            storage=storage,
        )
