"""Postgres-backed stores for tickets, runs, approvals, escalations and SLA policies.

One class implements the small store protocols used elsewhere (TicketStore, RunStore,
ApprovalStore, EscalationStore, SlaPolicyStore), so the API can swap the in-memory versions for
this one with a single setting. Queries are short and indexed; a sync pool keeps it simple.
"""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from relay.agent.runner import RunRecord
from relay.approvals import Approval, ApprovalError, ApprovalStatus, Decision, apply_decision
from relay.db import models
from relay.domain import Customer, Priority, Ticket, TicketStatus
from relay.escalation import Escalation, EscalationStatus
from relay.sla import SlaPolicy


class PostgresStore:
    def __init__(self, database_url: str, *, min_size: int = 1, max_size: int = 5) -> None:
        self.pool = ConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=True,
        )

    def close(self) -> None:
        self.pool.close()

    def _one(self, sql: str, params: Any = None) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchone()

    def _all(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    def _exec(self, sql: str, params: Any = None) -> None:
        with self.pool.connection() as conn:
            conn.execute(sql, params)

    # Tickets and customers ---------------------------------------------------------------

    def put_customer(self, customer: Customer) -> None:
        self._exec(
            "INSERT INTO customers (customer_id, name, email, plan) "
            "VALUES (%(customer_id)s, %(name)s, %(email)s, %(plan)s) "
            "ON CONFLICT (customer_id) DO UPDATE SET name = EXCLUDED.name, "
            "email = EXCLUDED.email, plan = EXCLUDED.plan",
            customer.model_dump(),
        )

    def get_customer(self, customer_id: str) -> Customer | None:
        row = self._one("SELECT * FROM customers WHERE customer_id = %s", (customer_id,))
        return models.customer_from_row(row) if row else None

    def put_ticket(self, ticket: Ticket) -> None:
        self._exec(
            f"INSERT INTO tickets ({models.TICKET_COLUMNS}) VALUES (%(ticket_id)s, "
            "%(customer_id)s, %(subject)s, %(body)s, %(priority)s, %(severity)s, %(channel)s, "
            "%(status)s, %(created_at)s, %(messages)s) "
            "ON CONFLICT (ticket_id) DO UPDATE SET subject = EXCLUDED.subject, "
            "body = EXCLUDED.body, priority = EXCLUDED.priority, severity = EXCLUDED.severity, "
            "status = EXCLUDED.status, messages = EXCLUDED.messages",
            models.ticket_params(ticket),
        )

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        row = self._one(
            f"SELECT {models.TICKET_COLUMNS} FROM tickets WHERE ticket_id = %s", (ticket_id,)
        )
        return models.ticket_from_row(row) if row else None

    def set_status(self, ticket_id: str, status: TicketStatus) -> None:
        self._exec("UPDATE tickets SET status = %s WHERE ticket_id = %s", (str(status), ticket_id))

    def list_tickets(self) -> list[Ticket]:
        rows = self._all(f"SELECT {models.TICKET_COLUMNS} FROM tickets ORDER BY created_at")
        return [models.ticket_from_row(r) for r in rows]

    # Runs --------------------------------------------------------------------------------

    def save_run(self, run: RunRecord) -> RunRecord:
        data = run.model_dump()
        data["status"] = str(data["status"])
        data["outcome"] = Jsonb(data["outcome"]) if data["outcome"] is not None else None
        self._exec(
            f"INSERT INTO runs ({models.RUN_COLUMNS}) VALUES (%(run_id)s, %(ticket_id)s, "
            "%(status)s, %(outcome)s, %(pending_approval_id)s, %(error)s, %(created_at)s, "
            "%(updated_at)s) ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status, "
            "outcome = EXCLUDED.outcome, pending_approval_id = EXCLUDED.pending_approval_id, "
            "error = EXCLUDED.error, updated_at = EXCLUDED.updated_at",
            data,
        )
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._one(f"SELECT {models.RUN_COLUMNS} FROM runs WHERE run_id = %s", (run_id,))
        return RunRecord.model_validate(row) if row else None

    def list_runs(self, ticket_id: str | None = None) -> list[RunRecord]:
        if ticket_id:
            rows = self._all(
                f"SELECT {models.RUN_COLUMNS} FROM runs WHERE ticket_id = %s ORDER BY created_at",
                (ticket_id,),
            )
        else:
            rows = self._all(f"SELECT {models.RUN_COLUMNS} FROM runs ORDER BY created_at")
        return [RunRecord.model_validate(r) for r in rows]

    # Approvals ---------------------------------------------------------------------------

    def create(self, approval: Approval) -> Approval:
        """Insert a pending approval; if one exists for the same key, return that one."""
        self._exec(
            f"INSERT INTO pending_approvals ({models.APPROVAL_COLUMNS}) VALUES (%(approval_id)s, "
            "%(key)s, %(run_id)s, %(ticket_id)s, %(tool)s, %(args)s, %(status)s, "
            "%(requested_at)s, %(decided_at)s, %(reviewer)s, %(note)s, %(edited_args)s) "
            "ON CONFLICT (key) DO NOTHING",
            models.approval_params(approval),
        )
        row = self._one(
            f"SELECT {models.APPROVAL_COLUMNS} FROM pending_approvals WHERE key = %s",
            (approval.key,),
        )
        return models.approval_from_row(row)

    def get(self, approval_id: str) -> Approval | None:
        row = self._one(
            f"SELECT {models.APPROVAL_COLUMNS} FROM pending_approvals WHERE approval_id = %s",
            (approval_id,),
        )
        return models.approval_from_row(row) if row else None

    def list(self, status: ApprovalStatus | None = None) -> list[Approval]:
        sql = f"SELECT {models.APPROVAL_COLUMNS} FROM pending_approvals"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = %s"
            params = (str(status),)
        rows = self._all(sql + " ORDER BY requested_at", params)
        return [models.approval_from_row(r) for r in rows]

    def decide(self, approval_id: str, decision: Decision) -> Approval:
        current = self.get(approval_id)
        if current is None:
            raise ApprovalError(f"unknown approval {approval_id}")
        decided = apply_decision(current, decision)
        params = models.approval_params(decided)
        with self.pool.connection() as conn:
            row = conn.execute(
                "UPDATE pending_approvals SET status = %(status)s, decided_at = %(decided_at)s, "
                "reviewer = %(reviewer)s, note = %(note)s, edited_args = %(edited_args)s "
                "WHERE approval_id = %(approval_id)s AND status = 'pending' RETURNING approval_id",
                params,
            ).fetchone()
        if row is None:
            raise ApprovalError(f"approval {approval_id} is already {current.status}")
        return decided

    # Escalations -------------------------------------------------------------------------

    def save(self, escalation: Escalation) -> Escalation:
        self._exec(
            f"INSERT INTO escalations ({models.ESCALATION_COLUMNS}) VALUES (%(escalation_id)s, "
            "%(ticket_id)s, %(run_id)s, %(reason)s, %(summary)s, %(trajectory)s, %(status)s, "
            "%(created_at)s, %(resolved_at)s, %(resolved_by)s, %(resolution)s, "
            "%(agent_should)s, %(expected_reason)s) ON CONFLICT (escalation_id) DO UPDATE SET "
            "status = EXCLUDED.status, resolved_at = EXCLUDED.resolved_at, "
            "resolved_by = EXCLUDED.resolved_by, resolution = EXCLUDED.resolution, "
            "agent_should = EXCLUDED.agent_should, expected_reason = EXCLUDED.expected_reason",
            models.escalation_params(escalation),
        )
        return escalation

    def get_escalation(self, escalation_id: str) -> Escalation | None:
        row = self._one(
            f"SELECT {models.ESCALATION_COLUMNS} FROM escalations WHERE escalation_id = %s",
            (escalation_id,),
        )
        return models.escalation_from_row(row) if row else None

    def list_escalations(self, status: EscalationStatus | None = None) -> list[Escalation]:
        sql = f"SELECT {models.ESCALATION_COLUMNS} FROM escalations"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = %s"
            params = (str(status),)
        rows = self._all(sql + " ORDER BY created_at", params)
        return [models.escalation_from_row(r) for r in rows]

    # SLA ---------------------------------------------------------------------------------

    def policies(self) -> dict[Priority, SlaPolicy]:
        rows = self._all(
            "SELECT priority, first_response_minutes, resolution_minutes FROM sla_policies"
        )
        return {Priority(r["priority"]): models.sla_from_row(r) for r in rows}


class PostgresEscalationStore:
    """Adapter giving PostgresStore the EscalationStore method names."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def add(self, escalation: Escalation) -> Escalation:
        return self._store.save(escalation)

    def save(self, escalation: Escalation) -> Escalation:
        return self._store.save(escalation)

    def get(self, escalation_id: str) -> Escalation | None:
        return self._store.get_escalation(escalation_id)

    def list(self, status: EscalationStatus | None = None) -> list[Escalation]:
        return self._store.list_escalations(status)
