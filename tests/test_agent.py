import asyncio

from relay.agent import AgentDeps, LocalToolClient, build_graph, graph_config, initial_state
from relay.config import Settings
from relay.domain import Customer, Ticket
from relay.providers import StubProvider
from relay.tools import ToolRegistry

from .conftest import NOW, make_world


def run(world, ticket_id="T-100", **settings_overrides):
    settings = Settings(_env_file=None, approval_tools=[], **settings_overrides)
    registry = ToolRegistry(world.context(), approval_tools=settings.approval_tools)
    deps = AgentDeps(StubProvider(), LocalToolClient(registry), settings, clock=world.clock)
    graph = build_graph(deps)
    ticket = world.tickets.get_ticket(ticket_id).for_model()
    return asyncio.run(graph.ainvoke(initial_state("run-1", ticket), graph_config("run-1", deps)))


def tool_steps(state):
    return [s for s in state["trajectory"] if s["type"] == "tool_call"]


def test_resolves_webhook_ticket_after_fixing_log_query():
    world = make_world()
    state = run(world)
    tools = [(s["tool"], s["error"]) for s in tool_steps(state)]
    assert tools == [
        ("recall_customer_history", None),
        ("lookup_ticket", None),
        ("search_docs", None),
        ("query_logs", "invalid_arguments"),
        ("query_logs", "invalid_query"),
        ("query_logs", None),
        ("draft_reply", None),
    ]
    assert [s["type"] for s in state["trajectory"]].count("replan") == 1
    assert state["outcome"]["status"] == "resolved"
    reply = world.outbox.sent[0]["body"]
    assert "req_a1" in reply and "KB-101" in reply
    assert "re-enabled" in reply  # recalled from the customer's memory


def test_step_limit_hands_off_with_gave_up():
    world = make_world()
    state = run(world, max_steps=2)
    assert state["outcome"]["status"] == "escalated"
    assert state["outcome"]["reason"] == "gave_up"
    assert tool_steps(state)[-1]["actor"] == "system"
    assert world.escalations.items[0]["reason"] == "gave_up"


def test_refund_request_is_escalated_with_reason():
    world = make_world()
    world.add_ticket(
        Ticket(
            ticket_id="T-200",
            customer_id="C-9",
            subject="Refund for duplicate charge",
            body="I was charged twice this month and want my money back.",
            created_at=NOW,
        ),
        Customer(customer_id="C-9", name="Sam Lee"),
    )
    state = run(world, "T-200")
    assert state["outcome"]["status"] == "escalated"
    assert state["outcome"]["reason"] == "needs_refund"
    assert world.outbox.sent == []
