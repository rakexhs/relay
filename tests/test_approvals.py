import asyncio
from datetime import timedelta

import pytest

from relay import sla
from relay.agent import AgentDeps, LocalToolClient
from relay.agent.runner import AgentRunner, RunStatus
from relay.approvals import ApprovalError, ApprovalStatus, Decision, InMemoryApprovalStore
from relay.config import Settings
from relay.domain import Priority, Severity, Ticket, TicketStatus
from relay.escalation import (
    EscalationService,
    InMemoryEscalationStore,
    Resolution,
    to_eval_task,
)
from relay.providers import StubProvider
from relay.tools import ToolRegistry

from .conftest import NOW, make_world


def make_runner(world=None):
    world = world or make_world()
    escalations = EscalationService(InMemoryEscalationStore())
    world.escalations = escalations
    settings = Settings(_env_file=None)
    registry = ToolRegistry(world.context(), approval_tools=settings.approval_tools)
    approvals = InMemoryApprovalStore()
    deps = AgentDeps(
        StubProvider(), LocalToolClient(registry), settings, clock=world.clock, approvals=approvals
    )
    return AgentRunner(deps, tickets=world.tickets), world, approvals, escalations


def test_reply_waits_for_approval_then_sends():
    runner, world, approvals, _ = make_runner()
    run = asyncio.run(runner.start(world.tickets.get_ticket("T-100"), run_id="run-a"))
    assert run.status == RunStatus.AWAITING_APPROVAL
    assert world.outbox.sent == []
    assert world.tickets.get_ticket("T-100").status == TicketStatus.AWAITING_APPROVAL
    pending = approvals.list(ApprovalStatus.PENDING)
    assert [a.tool for a in pending] == ["draft_reply"]

    run = asyncio.run(
        runner.decide(pending[0].approval_id, Decision(approved=True, reviewer="kim"))
    )
    assert run.status == RunStatus.COMPLETED
    assert run.outcome["status"] == "resolved"
    assert len(world.outbox.sent) == 1
    assert len(approvals.list()) == 1  # resuming did not create a second approval
    assert world.tickets.get_ticket("T-100").status == TicketStatus.RESOLVED


def test_reviewer_edits_are_what_gets_sent():
    runner, world, approvals, _ = make_runner()
    run = asyncio.run(runner.start(world.tickets.get_ticket("T-100"), run_id="run-b"))
    edited = dict(approvals.get(run.pending_approval_id).args, body="Hi Dana, edited reply text.")
    asyncio.run(runner.decide(run.pending_approval_id, Decision(approved=True, edited_args=edited)))
    assert world.outbox.sent[0]["body"] == "Hi Dana, edited reply text."


def test_rejected_reply_turns_into_an_escalation():
    runner, world, approvals, escalations = make_runner()
    run = asyncio.run(runner.start(world.tickets.get_ticket("T-100"), run_id="run-c"))
    run = asyncio.run(
        runner.decide(run.pending_approval_id, Decision(approved=False, note="wrong article"))
    )
    assert run.status == RunStatus.AWAITING_APPROVAL
    escalation_approval = approvals.get(run.pending_approval_id)
    assert escalation_approval.tool == "escalate_to_human"
    assert escalation_approval.args["reason"] == "low_confidence"

    run = asyncio.run(runner.decide(run.pending_approval_id, Decision(approved=True)))
    assert run.outcome == {**run.outcome, "status": "escalated", "reason": "low_confidence"}
    escalation = escalations.store.list()[0]
    assert any(s.get("type") == "approval_decision" for s in escalation.trajectory)
    assert world.outbox.sent == []


def test_cannot_decide_twice():
    runner, world, approvals, _ = make_runner()
    run = asyncio.run(runner.start(world.tickets.get_ticket("T-100"), run_id="run-d"))
    asyncio.run(runner.decide(run.pending_approval_id, Decision(approved=True)))
    with pytest.raises(ApprovalError):
        asyncio.run(runner.decide(run.pending_approval_id, Decision(approved=True)))


def test_breached_sla_escalates_before_any_model_call():
    world = make_world()
    old = world.tickets.get_ticket("T-100").model_copy(
        update={"priority": Priority.URGENT, "created_at": NOW - timedelta(hours=5)}
    )
    world.add_ticket(old)
    runner, world, _, escalations = make_runner(world)
    run = asyncio.run(runner.start(old, run_id="run-e"))
    assert run.status == RunStatus.COMPLETED
    assert run.outcome["reason"] == "sla_breach"
    assert escalations.store.list()[0].reason == "sla_breach"


def test_sla_ranking_puts_breached_and_critical_first():
    def ticket(tid, priority, severity, minutes_ago):
        return Ticket(
            ticket_id=tid,
            customer_id="C-1",
            subject="s",
            body="b",
            priority=priority,
            severity=severity,
            created_at=NOW - timedelta(minutes=minutes_ago),
        )

    tickets = [
        ticket("T-1", Priority.LOW, Severity.MINOR, 10),
        ticket("T-2", Priority.URGENT, Severity.CRITICAL, 300),  # breached (240m SLA)
        ticket("T-3", Priority.HIGH, Severity.MAJOR, 1000),
        ticket("T-4", Priority.URGENT, Severity.MAJOR, 10),
    ]
    ranked = sla.rank(tickets, sla.DEFAULT_POLICIES, NOW)
    assert [s.ticket_id for s in ranked] == ["T-2", "T-4", "T-3", "T-1"]
    assert [s.ticket_id for s in sla.breached(tickets, sla.DEFAULT_POLICIES, NOW)] == ["T-2"]


def test_resolved_escalation_exports_as_eval_task():
    runner, world, approvals, escalations = make_runner()
    record = escalations.escalate("T-100", "low_confidence", "Could not find a doc.", "run-x", [])
    resolved = escalations.resolve(
        record["escalation_id"],
        Resolution(
            resolution="Pointed them at KB-101 and re-enabled the endpoint.",
            resolved_by="kim",
            agent_should="resolve",
        ),
    )
    task = to_eval_task(resolved, world.tickets.get_ticket("T-100"))
    assert task["expect"]["outcome"] == "resolved"
    assert task["ticket"]["ticket_id"] == "T-100"
    assert task["reference_reply"].startswith("Pointed")
