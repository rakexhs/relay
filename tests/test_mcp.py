import asyncio

from mcp.shared.memory import create_connected_server_and_client_session

from relay.agent import AgentDeps
from relay.agent.runner import AgentRunner, RunStatus
from relay.approvals import Decision, InMemoryApprovalStore
from relay.config import Settings
from relay.mcp_client import MCPToolClient, in_process_client
from relay.mcp_server import build_mcp_server
from relay.providers import StubProvider
from relay.tools import CallMeta, ToolRegistry

from .conftest import make_world

APPROVAL_TOOLS = ["draft_reply", "escalate_to_human"]


def with_client(fn, world=None, approvals=None):
    world = world or make_world()
    approvals = approvals if approvals is not None else InMemoryApprovalStore()
    registry = ToolRegistry(world.context(), approval_tools=APPROVAL_TOOLS)

    async def go():
        async with in_process_client(registry, approvals, APPROVAL_TOOLS) as client:
            return await fn(client, registry, approvals, world)

    return asyncio.run(go())


def test_server_lists_the_six_tools_with_registry_schemas():
    async def check(client, registry, approvals, world):
        specs = {s.name: s for s in await client.list_tools()}
        assert set(specs) == set(registry.tools)
        for name, tool in registry.tools.items():
            assert specs[name].parameters == tool.spec().parameters
        return True

    assert with_client(check)


def test_structured_errors_survive_the_mcp_round_trip():
    async def check(client, *_):
        return await client.call("query_logs", {"query": "status:502"}, CallMeta())

    result = with_client(check)
    assert result["error"]["code"] == "invalid_query"
    assert "status_code" in result["error"]["allowed"]


def test_unapproved_send_over_mcp_becomes_a_pending_approval():
    async def check(client, registry, approvals, world):
        args = {"ticket_id": "T-100", "body": "Hi Dana, the fix is in KB-101.", "send": True}
        first = await client.call("draft_reply", args, CallMeta())
        assert first["status"] == "pending_approval"
        assert world.outbox.sent == []
        # A client claiming approval without a real approved record gets nowhere.
        forged = await client.call("draft_reply", args, CallMeta(approved=True))
        assert forged["status"] == "pending_approval"
        approvals.decide(first["approval_id"], Decision(approved=True, reviewer="kim"))
        sent = await client.call("draft_reply", args, CallMeta(approval_id=first["approval_id"]))
        assert sent["status"] == "sent"
        return len(world.outbox.sent)

    assert with_client(check) == 1


def test_agent_runs_end_to_end_through_mcp():
    async def check(client, registry, approvals, world):
        settings = Settings(_env_file=None)
        deps = AgentDeps(StubProvider(), client, settings, clock=world.clock, approvals=approvals)
        runner = AgentRunner(deps, tickets=world.tickets)
        run = await runner.start(world.tickets.get_ticket("T-100"), run_id="run-mcp")
        assert run.status == RunStatus.AWAITING_APPROVAL
        run = await runner.decide(run.pending_approval_id, Decision(approved=True))
        return run

    run = with_client(check)
    assert run.status == RunStatus.COMPLETED
    assert run.outcome["status"] == "resolved"


def test_raw_session_sees_json_results():
    world = make_world()
    registry = ToolRegistry(world.context())
    server = build_mcp_server(registry)

    async def go():
        async with create_connected_server_and_client_session(server) as session:
            client = MCPToolClient(session, [])
            return await client.call("lookup_ticket", {"ticket_id": "T-100"}, CallMeta())

    result = asyncio.run(go())
    assert result["customer"]["name"] == "Dana Ortiz"


def test_policy_hand_off_works_when_tools_are_behind_mcp():
    from datetime import timedelta

    from relay.domain import Priority
    from relay.escalation import EscalationService, InMemoryEscalationStore

    from .conftest import NOW

    world = make_world()
    old = world.tickets.get_ticket("T-100").model_copy(
        update={"priority": Priority.URGENT, "created_at": NOW - timedelta(hours=5)}
    )
    world.add_ticket(old)
    escalations = EscalationService(InMemoryEscalationStore())

    async def check(client, registry, approvals, world):
        deps = AgentDeps(
            StubProvider(),
            client,
            Settings(_env_file=None),
            clock=world.clock,
            approvals=approvals,
            escalations=escalations,
        )
        return await AgentRunner(deps).start(old, run_id="run-sla-mcp")

    run = with_client(check, world=world)
    assert run.outcome["status"] == "escalated"
    assert run.outcome["reason"] == "sla_breach"
    assert escalations.store.list()[0].run_id == "run-sla-mcp"
