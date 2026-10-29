-- Core tables for tickets, runs, approvals, escalations and SLA policies.
-- LangGraph's checkpoint tables are created separately by AsyncPostgresSaver.setup().

CREATE TABLE IF NOT EXISTS customers (
    customer_id  text PRIMARY KEY,
    name         text NOT NULL,
    email        text NOT NULL DEFAULT '',
    plan         text NOT NULL DEFAULT 'standard'
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id    text PRIMARY KEY,
    customer_id  text NOT NULL,
    subject      text NOT NULL,
    body         text NOT NULL,
    priority     text NOT NULL CHECK (priority IN ('urgent', 'high', 'normal', 'low')),
    severity     text NOT NULL CHECK (severity IN ('critical', 'major', 'minor')),
    channel      text NOT NULL,
    status       text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    messages     jsonb NOT NULL DEFAULT '[]'::jsonb
);
CREATE INDEX IF NOT EXISTS tickets_open_idx ON tickets (status, created_at)
    WHERE status IN ('open', 'in_progress', 'awaiting_approval');

CREATE TABLE IF NOT EXISTS sla_policies (
    priority                text PRIMARY KEY,
    first_response_minutes  integer NOT NULL,
    resolution_minutes      integer NOT NULL
);
INSERT INTO sla_policies (priority, first_response_minutes, resolution_minutes) VALUES
    ('urgent', 30, 240),
    ('high', 120, 1440),
    ('normal', 480, 2880),
    ('low', 1440, 7200)
ON CONFLICT (priority) DO NOTHING;

CREATE TABLE IF NOT EXISTS runs (
    run_id               text PRIMARY KEY,
    ticket_id            text NOT NULL,
    status               text NOT NULL,
    outcome              jsonb,
    pending_approval_id  text,
    error                text,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS runs_ticket_idx ON runs (ticket_id);

CREATE TABLE IF NOT EXISTS pending_approvals (
    approval_id   text PRIMARY KEY,
    key           text NOT NULL UNIQUE,
    run_id        text NOT NULL,
    ticket_id     text NOT NULL,
    tool          text NOT NULL,
    args          jsonb NOT NULL,
    status        text NOT NULL DEFAULT 'pending',
    requested_at  timestamptz NOT NULL DEFAULT now(),
    decided_at    timestamptz,
    reviewer      text,
    note          text,
    edited_args   jsonb
);
CREATE INDEX IF NOT EXISTS pending_approvals_status_idx ON pending_approvals (status, requested_at);

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id    text PRIMARY KEY,
    ticket_id        text NOT NULL,
    run_id           text,
    reason           text NOT NULL CHECK (reason IN ('sla_breach', 'low_confidence', 'needs_refund',
                                                     'tool_failure', 'customer_request', 'gave_up')),
    summary          text NOT NULL,
    trajectory       jsonb NOT NULL DEFAULT '[]'::jsonb,
    status           text NOT NULL DEFAULT 'open',
    created_at       timestamptz NOT NULL DEFAULT now(),
    resolved_at      timestamptz,
    resolved_by      text,
    resolution       text,
    agent_should     text CHECK (agent_should IN ('resolve', 'escalate')),
    expected_reason  text
);
CREATE INDEX IF NOT EXISTS escalations_status_idx ON escalations (status, created_at);
