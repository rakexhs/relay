"""The LangGraph wiring for the support agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from relay.agent.nodes import AgentDeps, AgentNodes
from relay.agent.state import AgentState


def _after_intake(state: AgentState) -> str:
    return "finish" if state.get("route") == "finish" else "plan"


def _after_approval(state: AgentState) -> str:
    return "execute" if state.get("pending") else "observe"


def _after_observe(state: AgentState) -> str:
    route = state.get("route", "act")
    return "plan" if route == "replan" else route


def build_graph(deps: AgentDeps, checkpointer: Any = None):
    nodes = AgentNodes(deps)

    def after_act(state: AgentState) -> str:
        if not state.get("pending"):
            return "finish"
        return "request_approval" if nodes.needs_approval(state) else "execute"

    graph = StateGraph(AgentState)
    graph.add_node("intake", nodes.intake)
    graph.add_node("plan", nodes.plan)
    graph.add_node("act", nodes.act)
    graph.add_node("request_approval", nodes.request_approval)
    graph.add_node("await_approval", nodes.await_approval)
    graph.add_node("execute", nodes.execute)
    graph.add_node("observe", nodes.observe)
    graph.add_node("finish", nodes.finish)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", _after_intake, ["plan", "finish"])
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", after_act, ["request_approval", "execute", "finish"])
    graph.add_edge("request_approval", "await_approval")
    graph.add_conditional_edges("await_approval", _after_approval, ["execute", "observe"])
    graph.add_edge("execute", "observe")
    graph.add_conditional_edges("observe", _after_observe, ["act", "plan", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(
    run_id: str, ticket: dict[str, Any], sla_deadline: str | None = None
) -> AgentState:
    return {
        "run_id": run_id,
        "ticket": ticket,
        "sla_deadline": sla_deadline,
        "approval": None,
        "stop_reason": None,
        "messages": [],
        "trajectory": [],
        "plan": [],
        "replans": 0,
        "pending": None,
        "tool_calls": 0,
        "consecutive_errors": 0,
        "outcome": None,
    }


# Commit each super-step's checkpoint before the next step starts. LangGraph's default
# ("async") overlaps the write with the next step, so a SIGKILL could lose the last steps.
DURABILITY = "sync"


def graph_config(run_id: str, deps: AgentDeps) -> dict[str, Any]:
    # Each node visit is a super-step; leave room for plan/act/execute/observe per tool call.
    limit = deps.settings.max_steps * 4 + 20
    return {"configurable": {"thread_id": run_id}, "recursion_limit": limit}
