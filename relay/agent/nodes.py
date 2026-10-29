"""Graph nodes.

    intake -> plan -> act -> [request_approval -> await_approval] -> execute -> observe
    observe -> act | plan (replan) | finish

Each node does one thing and returns a partial state update. `execute` runs exactly one tool
call, so every completed tool call ends a LangGraph super-step and gets its own checkpoint.
Write tools pass through two nodes first: `request_approval` stores the pending approval (its
key makes it idempotent), and `await_approval` does nothing but interrupt. On resume LangGraph
re-runs only the interrupted node, so nothing is written twice.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langgraph.types import interrupt

from relay import tracing
from relay.agent.client import ToolClient
from relay.agent.prompts import (
    PLAN_INSTRUCTION,
    PROCEED_INSTRUCTION,
    REPLAN_INSTRUCTION,
    SYSTEM_PROMPT,
    parse_plan,
    render_observation,
    render_ticket,
)
from relay.agent.state import AgentState
from relay.approvals import Approval, ApprovalStore, Decision, InMemoryApprovalStore
from relay.config import Settings
from relay.domain import utcnow
from relay.providers.base import Message, ModelProvider, ModelRequest, ModelResponse, ToolCall
from relay.tools.backends import EscalationSink
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
    approvals: ApprovalStore = field(default_factory=InMemoryApprovalStore)
    # Where policy-driven hand-offs go; None means "through the escalate_to_human tool".
    escalations: EscalationSink | None = None


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
        provider = deps.provider
        self._model_call = tracing.maybe_traceable(
            provider.complete, name=f"{provider.name}:{provider.model}", run_type="llm"
        )

    async def _complete(self, request: ModelRequest) -> ModelResponse:
        provider = self.deps.provider
        with tracing.model_span(provider.name, provider.model, request.purpose) as span:
            response = await self._model_call(request)
            for key, value in response.usage.items():
                span.set_attribute(f"relay.model.{key}", value)
            return response

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
        approval_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        meta = CallMeta(
            run_id=state.get("run_id"),
            trajectory=list(state.get("trajectory", [])),
            approved=approved,
            approval_id=approval_id,
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

    def _sla_breached(self, state: AgentState) -> bool:
        deadline = state.get("sla_deadline")
        return bool(deadline) and self.deps.clock() >= datetime.fromisoformat(deadline)

    async def intake(self, state: AgentState) -> dict[str, Any]:
        """Recall the customer's history and open the conversation with the ticket."""
        ticket = state["ticket"]
        result, step = await self._run_tool(
            state,
            "recall_customer_history",
            {
                "customer_id": ticket["customer_id"],
                "query": ticket["subject"],
                "top_k": self.deps.settings.memory_top_k,
            },
            actor="system",
        )
        memories = [] if is_error(result) else result.get("memories", [])
        content, reports = render_ticket(ticket, memories, quote=self.deps.settings.quote_retrieved)
        flagged = [r.as_dict() for r in reports if r.flagged]
        if flagged:
            step["safety"] = flagged[0]
        first = Message(role="user", content=content)
        update: dict[str, Any] = {"messages": [first.model_dump()], "trajectory": [step]}
        if self._sla_breached(state):
            update["route"] = "finish"
            update["stop_reason"] = "sla_breach"
        return update

    async def plan(self, state: AgentState) -> dict[str, Any]:
        """Ask for a plan and keep the exchange in the transcript.

        The history only ever grows: the system prompt never changes and the plan is a normal
        assistant turn, followed by a user turn telling the model to proceed. Providers that
        cache prompts or bind reasoning to the exact history keep working across replans.
        """
        replanning = state.get("route") == "replan"
        ask = Message(role="user", content=REPLAN_INSTRUCTION if replanning else PLAN_INSTRUCTION)
        messages = [Message(**m) for m in state["messages"]] + [ask]
        response = await self._complete(
            ModelRequest(purpose="plan", system=SYSTEM_PROMPT, messages=messages)
        )
        steps = parse_plan(response.text)
        answer = Message(
            role="assistant", content=response.text, provider_content=response.provider_content
        )
        proceed = Message(role="user", content=PROCEED_INSTRUCTION)
        record = {"type": "replan" if replanning else "plan", "at": self._now(), "steps": steps}
        update: dict[str, Any] = {
            "plan": steps,
            "trajectory": [record],
            "route": "act",
            "messages": [ask.model_dump(), answer.model_dump(), proceed.model_dump()],
        }
        if replanning:
            update["replans"] = state.get("replans", 0) + 1
        return update

    async def act(self, state: AgentState) -> dict[str, Any]:
        """Ask the model for the next tool call (or a final answer)."""
        response = await self._complete(
            ModelRequest(
                purpose="act",
                system=SYSTEM_PROMPT,
                messages=[Message(**m) for m in state["messages"]],
                tools=await self._tool_specs(),
            )
        )
        if not response.tool_calls:
            return {"pending": None, "final_text": response.text}
        call: ToolCall = response.tool_calls[0]
        # Replaying raw provider content is only safe when it holds exactly the call we run.
        raw = response.provider_content if len(response.tool_calls) == 1 else None
        assistant = Message(
            role="assistant", content=response.text, tool_calls=[call], provider_content=raw
        )
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

    def needs_approval(self, state: AgentState) -> bool:
        pending = state.get("pending")
        if not pending or pending.get("approved"):
            return False
        return self.deps.tools.requires_approval(pending["name"], pending["args"])

    async def request_approval(self, state: AgentState) -> dict[str, Any]:
        call = state["pending"]
        approval = self.deps.approvals.create(
            Approval(
                key=f"{state['run_id']}:{call['id']}",
                run_id=state["run_id"],
                ticket_id=state["ticket"]["ticket_id"],
                tool=call["name"],
                args=call["args"],
            )
        )
        record = {
            "type": "approval_requested",
            "at": self._now(),
            "approval_id": approval.approval_id,
            "tool": call["name"],
            "args": call["args"],
        }
        return {"approval": approval.model_dump(mode="json"), "trajectory": [record]}

    async def await_approval(self, state: AgentState) -> dict[str, Any]:
        approval = state["approval"]
        raw = interrupt(
            {
                "approval_id": approval["approval_id"],
                "tool": approval["tool"],
                "args": approval["args"],
            }
        )
        decision = Decision.model_validate(raw)
        record = {
            "type": "approval_decision",
            "at": self._now(),
            "approval_id": approval["approval_id"],
            "approved": decision.approved,
            "reviewer": decision.reviewer,
            "note": decision.note,
            "edited": decision.edited_args is not None,
        }
        pending = dict(state["pending"])
        if decision.approved:
            pending["args"] = decision.edited_args or pending["args"]
            pending["approved"] = True
            pending["approval_id"] = approval["approval_id"]
            return {"pending": pending, "approval": None, "trajectory": [record]}
        # A rejection is the tool's answer: the model sees it and has to choose another path.
        result = {"status": "rejected", "note": decision.note or "The reviewer rejected this."}
        observation = Message(
            role="tool",
            tool_call_id=pending["id"],
            name=pending["name"],
            content=f"Result of {pending['name']}:\n{json.dumps(result)}",
        )
        return {
            "pending": None,
            "approval": None,
            "messages": [observation.model_dump()],
            "trajectory": [record],
            "tool_calls": state.get("tool_calls", 0) + 1,
        }

    async def execute(self, state: AgentState) -> dict[str, Any]:
        pending = dict(state["pending"])
        approved = bool(pending.pop("approved", False))
        approval_id = pending.pop("approval_id", None)
        call = ToolCall(**pending)
        result, step = await self._run_tool(
            state, call.name, call.args, actor="agent", approved=approved, approval_id=approval_id
        )
        if approved:
            step["approved"] = True
            step["approval_id"] = approval_id
        content, report = render_observation(
            call.name, result, quote=self.deps.settings.quote_retrieved
        )
        if report.flagged:
            step["safety"] = report.as_dict()
        observation = Message(role="tool", tool_call_id=call.id, name=call.name, content=content)
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
        if self._sla_breached(state):
            note = {"type": "note", "at": self._now(), "note": "SLA deadline passed"}
            return {"route": "finish", "stop_reason": "sla_breach", "trajectory": [note]}
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

    async def _system_escalation(
        self, state: AgentState, reason: str, summary: str
    ) -> dict[str, Any]:
        """Policy hand-offs (SLA breach, step limit) skip the model and the approval gate.

        They go straight to the escalation service when one is wired in, because the tool
        path may be a remote MCP server that rightly refuses unapproved writes.
        """
        args = {"ticket_id": state["ticket"]["ticket_id"], "reason": reason, "summary": summary}
        if self.deps.escalations is None:
            _, step = await self._run_tool(
                state, "escalate_to_human", args, actor="system", approved=True
            )
            return step
        record = self.deps.escalations.escalate(
            args["ticket_id"], reason, summary, state.get("run_id"), list(state["trajectory"])
        )
        return {
            "type": "tool_call",
            "at": self._now(),
            "actor": "system",
            "tool": "escalate_to_human",
            "args": args,
            "result": {
                "status": "escalated",
                "escalation_id": record["escalation_id"],
                "reason": reason,
            },
            "error": None,
            "latency_ms": 0.0,
            "via": "escalation_service",
        }

    async def finish(self, state: AgentState) -> dict[str, Any]:
        trajectory = list(state.get("trajectory", []))
        outcome = derive_outcome(trajectory)
        update: dict[str, Any] = {}
        if outcome is None:
            # Nothing resolved the ticket: hand it off so it never silently stalls.
            reason = state.get("stop_reason") or "gave_up"
            if reason == "sla_breach":
                summary = "SLA resolution deadline passed before the ticket was resolved."
            else:
                summary = (
                    f"Agent stopped after {state.get('tool_calls', 0)} tool calls without "
                    f"resolving the ticket. Last model note: {state.get('final_text') or 'none'}"
                )
            step = await self._system_escalation(state, reason, summary[:2000])
            trajectory.append(step)
            update["trajectory"] = [step]
            outcome = derive_outcome(trajectory) or {"status": "failed", "reason": reason}
        outcome = {**outcome, "tool_calls": state.get("tool_calls", 0)}
        record = {"type": "finish", "at": self._now(), "outcome": outcome}
        update["trajectory"] = [*update.get("trajectory", []), record]
        update["outcome"] = outcome
        return update
