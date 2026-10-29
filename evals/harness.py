"""Run the real agent graph over eval tasks and collect what happened."""

from __future__ import annotations

import time
from contextlib import AsyncExitStack
from typing import Any

from pydantic import BaseModel, Field

from evals.schema import Task
from evals.world import build_world
from relay.agent import AgentDeps, LocalToolClient
from relay.agent.runner import AgentRunner, RunStatus
from relay.approvals import Decision, InMemoryApprovalStore
from relay.config import Settings
from relay.escalation import EscalationService, InMemoryEscalationStore
from relay.providers import ModelProvider
from relay.tools import ToolRegistry

MAX_APPROVALS_PER_TASK = 10


class TaskResult(BaseModel):
    task_id: str
    category: str
    status: str  # completed | failed
    outcome: dict[str, Any] | None = None
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    sent: list[dict[str, Any]] = Field(default_factory=list)
    escalations: list[dict[str, Any]] = Field(default_factory=list)
    approvals: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    def calls(self) -> list[dict[str, Any]]:
        return [s for s in self.trajectory if s.get("type") == "tool_call"]


async def run_task(
    task: Task, provider: ModelProvider, settings: Settings, transport: str = "local"
) -> TaskResult:
    """Run one task end to end. A scripted reviewer approves every write, as in staging."""
    world, ticket = build_world(task)
    escalations = EscalationService(InMemoryEscalationStore())
    world.escalations = escalations
    approvals = InMemoryApprovalStore()
    registry = ToolRegistry(world.context(), approval_tools=settings.approval_tools)

    async with AsyncExitStack() as stack:
        if transport == "mcp":
            from relay.mcp_client import in_process_client

            client: Any = await stack.enter_async_context(
                in_process_client(registry, approvals, settings.approval_tools)
            )
        else:
            client = LocalToolClient(registry)
        deps = AgentDeps(
            provider,
            client,
            settings,
            clock=world.clock,
            approvals=approvals,
            escalations=escalations,
        )
        runner = AgentRunner(deps, tickets=world.tickets)
        run_id = f"eval-{task.id}"
        started = time.perf_counter()
        decisions = 0
        try:
            run = await runner.start(ticket, run_id=run_id)
            while run.status == RunStatus.AWAITING_APPROVAL and decisions < MAX_APPROVALS_PER_TASK:
                decisions += 1
                run = await runner.decide(
                    run.pending_approval_id, Decision(approved=True, reviewer="eval-harness")
                )
            error = None if run.status == RunStatus.COMPLETED else f"run ended as {run.status}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        snapshot = await runner.snapshot(run_id)

    return TaskResult(
        task_id=task.id,
        category=task.category,
        status="failed" if error else "completed",
        outcome=snapshot["outcome"],
        trajectory=snapshot["trajectory"],
        sent=list(world.outbox.sent),
        escalations=[e.model_dump(mode="json") for e in escalations.store.list()],
        approvals=decisions,
        tool_calls=snapshot["tool_calls"],
        latency_ms=round(latency_ms, 2),
        error=error,
    )


async def run_suite(
    tasks: list[Task], provider: ModelProvider, settings: Settings, transport: str = "local"
) -> list[TaskResult]:
    # Sequential on purpose: latency numbers stay per-ticket, and providers are not hammered.
    return [await run_task(task, provider, settings, transport) for task in tasks]
