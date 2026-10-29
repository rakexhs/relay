"""Checkpointer selection: Postgres in production, in-memory for tests and local runs.

With the Postgres saver every super-step is written before the next one starts. Because the
`execute` node runs exactly one tool call, a process killed mid-run resumes from the last
completed tool call: `AgentRunner.recover(run_id)` (or `graph.ainvoke(None, config)`) picks up
from that checkpoint, and only a call that was in flight when the process died runs again.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from relay.config import Settings


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[Any]:
    if settings.checkpointer == "postgres":
        if not settings.database_url:
            raise RuntimeError("RELAY_CHECKPOINTER=postgres needs RELAY_DATABASE_URL")
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
            await saver.setup()
            yield saver
    else:
        yield InMemorySaver()
