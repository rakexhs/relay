"""Subprocess half of the kill-and-resume test: start a run that hangs on a given tool call.

python -m tests.kill_resume_worker <database_url> <run_id> <log_path> <hang_at>
"""

from __future__ import annotations

import asyncio
import sys

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from relay.agent import AgentDeps, LocalToolClient, build_graph, graph_config, initial_state
from relay.config import Settings
from relay.providers import StubProvider
from relay.tools import ToolRegistry
from tests.conftest import make_world
from tests.recording_client import RecordingToolClient


def build(database_url_saver, log_path, hang_at):
    world = make_world()
    settings = Settings(_env_file=None, approval_tools=[])
    registry = ToolRegistry(world.context(), approval_tools=[])
    client = RecordingToolClient(LocalToolClient(registry), log_path, hang_at=hang_at)
    deps = AgentDeps(StubProvider(), client, settings, clock=world.clock)
    return world, deps, build_graph(deps, database_url_saver)


async def main(database_url: str, run_id: str, log_path: str, hang_at: int) -> None:
    async with AsyncPostgresSaver.from_conn_string(database_url) as saver:
        await saver.setup()
        world, deps, graph = build(saver, log_path, hang_at)
        ticket = world.tickets.get_ticket("T-100").for_model()
        await graph.ainvoke(initial_state(run_id, ticket), graph_config(run_id, deps))


if __name__ == "__main__":
    url, run_id, log_path, hang_at = sys.argv[1:5]
    asyncio.run(main(url, run_id, log_path, int(hang_at)))
