"""A run that dies mid-way resumes from its last completed tool call."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from relay.agent import AgentDeps, LocalToolClient
from relay.agent.runner import AgentRunner, RunStatus
from relay.config import Settings
from relay.providers import StubProvider
from relay.tools import ToolRegistry

from .conftest import make_world
from .recording_client import RecordingToolClient, SimulatedCrash, read_log

ROOT = Path(__file__).resolve().parent.parent
DATABASE_URL = os.environ.get("RELAY_TEST_DATABASE_URL", "")


def _key(event):
    return (event["tool"], json.dumps(event["args"], sort_keys=True))


def _assert_no_completed_call_ran_twice(events):
    done = [_key(e) for e in events if e["event"] == "done"]
    assert len(done) == len(set(done)), done


def test_crashed_run_recovers_from_checkpoint(tmp_path):
    log = tmp_path / "calls.jsonl"
    world = make_world()
    settings = Settings(_env_file=None, approval_tools=[])
    registry = ToolRegistry(world.context(), approval_tools=[])
    client = RecordingToolClient(LocalToolClient(registry), log, crash_at=5)
    runner = AgentRunner(AgentDeps(StubProvider(), client, settings, clock=world.clock))

    with pytest.raises(SimulatedCrash):
        asyncio.run(runner.start(world.tickets.get_ticket("T-100"), run_id="run-crash"))
    assert runner.runs.get_run("run-crash").status == RunStatus.FAILED
    before = read_log(log)
    assert [e["event"] for e in before].count("done") == 4

    run = asyncio.run(runner.recover("run-crash"))
    assert run.status == RunStatus.COMPLETED
    assert run.outcome["status"] == "resolved"
    events = read_log(log)
    _assert_no_completed_call_ran_twice(events)
    resumed = events[len(before) :]
    # The call that crashed is the first thing retried; nothing before it runs again.
    assert _key(resumed[0]) == _key(before[-1])


@pytest.mark.postgres
@pytest.mark.skipif(not DATABASE_URL, reason="set RELAY_TEST_DATABASE_URL to run")
def test_sigkilled_process_resumes_from_postgres_checkpoint(tmp_path):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from relay.agent import build_graph, graph_config

    log = tmp_path / "calls.jsonl"
    run_id = f"kill-{uuid.uuid4().hex[:8]}"
    hang_at = 5
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.kill_resume_worker",
            DATABASE_URL,
            run_id,
            str(log),
            str(hang_at),
        ],
        cwd=ROOT,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if any(e["event"] == "start" and e["n"] == hang_at for e in read_log(log)):
                break
            if proc.poll() is not None:
                pytest.fail(f"worker exited early with {proc.returncode}")
            time.sleep(0.1)
        else:
            pytest.fail("worker never reached the hanging tool call")
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()

    before = read_log(log)
    assert [e["event"] for e in before].count("done") == hang_at - 1

    async def resume():
        async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
            world = make_world()
            settings = Settings(_env_file=None, approval_tools=[])
            registry = ToolRegistry(world.context(), approval_tools=[])
            client = RecordingToolClient(LocalToolClient(registry), log)
            deps = AgentDeps(StubProvider(), client, settings, clock=world.clock)
            graph = build_graph(deps, saver)
            config = graph_config(run_id, deps)
            snapshot = await graph.aget_state(config)
            assert snapshot.values["tool_calls"] == hang_at - 2  # intake's recall is not counted
            return await graph.ainvoke(None, config)

    state = asyncio.run(resume())
    assert state["outcome"]["status"] == "resolved"
    events = read_log(log)
    _assert_no_completed_call_ran_twice(events)
    resumed = events[len(before) :]
    assert _key(resumed[0]) == _key(before[-1])
