"""Graph nodes: intake -> plan -> act -> execute -> observe -> (act | replan | finish).

Each node does one thing and returns a partial state update. `execute` runs exactly one tool
call, so every completed tool call ends a LangGraph super-step and gets its own checkpoint.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from relay.agent.client import ToolClient
from relay.agent.prompts import (
    PLAN_INSTRUCTION,
    REPLAN_INSTRUCTION,
    parse_plan,
    render_observation,
    render_ticket,
    system_with_plan,
)
from relay.agent.state import AgentState
from relay.config import Settings
from relay.domain import utcnow
from relay.providers.base import Message, ModelProvider, ModelRequest, ToolCall
from relay.tools.errors import is_error
from relay.tools.registry import CallMeta

REPLAN_AFTER_ERRORS = 2
MAX_REPLANS = 2


@dataclass
class AgentDeps:
    provider: ModelProvider
    tools: ToolClient
    settings: Settings
    clock: Callable[[], datetime] = utcnow


def derive_outcome(trajectory: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolved if a reply was sent, escalated if a hand-off succeeded; the last one wins."""
    outcome = None
    for step in trajectory:
        if step.get("type") != "tool_call" or step.get("error"):
            continue
        result = step.get("result") or {}
        if step["tool"] == "escalate_to_human" and result.get("status") == "escalated":
            outcome = {
                "status": "escalated",
                "reason": step["args"].get("reason"),
                "escalation_id": result.get("escalation_id"),
                "actor": step.get("actor", "agent"),
            }
        elif step["tool"] == "draft_reply" and result.get("status") == "sent":
            outcome = {
                "status": "resolved",
                "reason": None,
                "reply": step["args"].get("body"),
                "message_id": result.get("message_id"),
                "actor": step.get("actor", "agent"),
            }
    return outcome


class AgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self._specs = None

    # helpers -----------------------------------------------------------------------------

    def _now(self) -> str:
        return self.deps.clock().isoformat()

    async def _tool_specs(self):
        if self._specs is None:
            self._specs = await self.deps.tools.list_tools()
        return self._specs

    async def _run_tool(
        self,
        state: AgentState,
        name: str,
        args: dict[str, Any],
        *,
        actor: str,
        approved: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        meta = CallMeta(
            run_id=state.get("run_id"),
            trajectory=list(state.get("trajectory", [])),
            approved=approved,
        )
        started = time.perf_counter()
        result = await self.deps.tools.call(name, args, meta)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        step = {
            "type": "tool_call",
            "at": self._now(),
            "actor": actor,
            "tool": name,
            "args": args,
            "result": result,
            "error": result["error"]["code"] if is_error(result) else None,
            "latency_ms": latency_ms,
            **(extra or {}),
        }
        return result, step

    # nodes -------------------------------------------------------------------------------

    async def intake(self, state: AgentState) -> dict[str, Any]:
        """Recall the customer's history and open the conversation with the ticket."""
        ticket = state["ticket"]
        result, step = await self._run_tool(
            state,
            "recall_customer_history",
            {"customer_id": ticket["customer_id"], "query": ticket["subject"], "top_k": 3},
            actor="system",
        )
        memories = [] if is_error(result) else result.get("memories", [])
        first = Message(role="user", content=render_ticket(ticket, memories))
        return {"messages": [first.model_dump()], "trajectory": [step]}

    async def plan(self, state: AgentState) -> dict[str, Any]:
        replanning = state.get("route") == "replan"
        instruction = REPLAN_INSTRUCTION if replanning else PLAN_INSTRUCTION
        messages = [Message(**m) for m in state["messages"]]
        messages.append(Message(role="user", content=instruction))
        response = await self.deps.provider.complete(
            ModelRequest(purpose="plan", system=system_with_plan([]), messages=messages)
        )
        steps = parse_plan(response.text)
        record = {"type": "replan" if replanning else "plan", "at": self._now(), "steps": steps}
        update: dict[str, Any] = {"plan": steps, "trajectory": [record], "route": "act"}
        if replanning:
            update["replans"] = state.get("replans", 0) + 1
        return update

    async def act(self, state: AgentState) -> dict[str, Any]:
        """Ask the model for the next tool call (or a final answer)."""
        response = await self.deps.provider.complete(
            ModelRequest(
                purpose="act",
                system=system_with_plan(state.get("plan", [])),
                messages=[Message(**m) for m in state["messages"]],
                tools=await self._tool_specs(),
            )
        )
        if not response.tool_calls:
            return {"pending": None, "final_text": response.text}
        call: ToolCall = response.tool_calls[0]
        assistant = Message(role="assistant", content=response.text, tool_calls=[call])
        update: dict[str, Any] = {
            "pending": call.model_dump(),
            "messages": [assistant.model_dump()],
        }
        if len(response.tool_calls) > 1:
            dropped = [c.name for c in response.tool_calls[1:]]
            update["trajectory"] = [
                {"type": "note", "at": self._now(), "note": f"ignored extra tool calls: {dropped}"}
            ]
        return update

    async def execute(self, state: AgentState) -> dict[str, Any]:
        call = ToolCall(**state["pending"])
        result, step = await self._run_tool(state, call.name, call.args, actor="agent")
        observation = Message(
            role="tool",
            tool_call_id=call.id,
            name=call.name,
            content=render_observation(call.name, result),
        )
        errors = state.get("consecutive_errors", 0) + 1 if step["error"] else 0
        return {
            "pending": None,
            "messages": [observation.model_dump()],
            "trajectory": [step],
            "tool_calls": state.get("tool_calls", 0) + 1,
            "consecutive_errors": errors,
        }

    async def observe(self, state: AgentState) -> dict[str, Any]:
        settings = self.deps.settings
        if derive_outcome(state.get("trajectory", [])) is not None:
            return {"route": "finish"}
        if state.get("tool_calls", 0) >= settings.max_steps:
            note = {"type": "note", "at": self._now(), "note": "step limit reached"}
            return {"route": "finish", "trajectory": [note]}
        errors = state.get("consecutive_errors", 0)
        if errors >= settings.max_consecutive_errors:
            note = {"type": "note", "at": self._now(), "note": f"{errors} errors in a row"}
            return {"route": "finish", "trajectory": [note]}
        if errors >= REPLAN_AFTER_ERRORS and state.get("replans", 0) < MAX_REPLANS:
            return {"route": "replan"}
        return {"route": "act"}

    async def finish(self, state: AgentState) -> dict[str, Any]:
        trajectory = list(state.get("trajectory", []))
        outcome = derive_outcome(trajectory)
        update: dict[str, Any] = {}
        if outcome is None:
            # Nothing resolved the ticket: hand it off so it never silently stalls.
            summary = (
                f"Agent stopped after {state.get('tool_calls', 0)} tool calls without resolving "
                f"the ticket. Last model note: {state.get('final_text') or 'none'}"
            )
            _, step = await self._run_tool(
                state,
                "escalate_to_human",
                {
                    "ticket_id": state["ticket"]["ticket_id"],
                    "reason": "gave_up",
                    "summary": summary[:2000],
                },
                actor="system",
                approved=True,
            )
            trajectory.append(step)
            update["trajectory"] = [step]
            outcome = derive_outcome(trajectory) or {"status": "failed", "reason": "gave_up"}
        outcome = {**outcome, "tool_calls": state.get("tool_calls", 0)}
        record = {"type": "finish", "at": self._now(), "outcome": outcome}
        update["trajectory"] = [*update.get("trajectory", []), record]
        update["outcome"] = outcome
        return update
