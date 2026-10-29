"""The LangGraph wiring for the support agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from relay.agent.nodes import AgentDeps, AgentNodes
from relay.agent.state import AgentState


def _after_act(state: AgentState) -> str:
    return "execute" if state.get("pending") else "finish"


def _after_observe(state: AgentState) -> str:
    route = state.get("route", "act")
    return "plan" if route == "replan" else route


def build_graph(deps: AgentDeps, checkpointer: Any = None):
    nodes = AgentNodes(deps)
    graph = StateGraph(AgentState)
    graph.add_node("intake", nodes.intake)
    graph.add_node("plan", nodes.plan)
    graph.add_node("act", nodes.act)
    graph.add_node("execute", nodes.execute)
    graph.add_node("observe", nodes.observe)
    graph.add_node("finish", nodes.finish)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", _after_act, ["execute", "finish"])
    graph.add_edge("execute", "observe")
    graph.add_conditional_edges("observe", _after_observe, ["act", "plan", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(run_id: str, ticket: dict[str, Any]) -> AgentState:
    return {
        "run_id": run_id,
        "ticket": ticket,
        "messages": [],
        "trajectory": [],
        "plan": [],
        "replans": 0,
        "pending": None,
        "tool_calls": 0,
        "consecutive_errors": 0,
        "outcome": None,
    }


def graph_config(run_id: str, deps: AgentDeps) -> dict[str, Any]:
    # Each node visit is a super-step; leave room for plan/act/execute/observe per tool call.
    limit = deps.settings.max_steps * 4 + 20
    return {"configurable": {"thread_id": run_id}, "recursion_limit": limit}
