# Architecture notes

This document covers how a run moves through the system, what is persisted where, and how
the evaluation reuses the production path. The README has the overview and quickstart.

## Life of a run

```text
POST /tickets/{id}/run
  AgentRunner.start(ticket)
    RunRecord(status=running) saved; ticket -> in_progress
    graph.ainvoke(initial_state, thread_id=run_id)
      intake            recall_customer_history (actor=system), SLA check
      plan              provider.complete(purpose="plan"); plan kept as transcript turns
      act               provider.complete(purpose="act", tools=specs)  -> pending call
      execute           ToolClient.call -> ToolRegistry.invoke -> handler
      observe           route: act | plan (replan) | finish
      ...
      act               model asks for draft_reply(send=true)
      request_approval  ApprovalStore.create(key=run_id:call_id)
      await_approval    interrupt({approval_id, tool, args})      <- graph pauses here
    RunRecord(status=awaiting_approval, pending_approval_id) saved; ticket -> awaiting_approval

POST /approvals/{id}  {"approved": true}
  AgentRunner.decide -> ApprovalStore.decide -> graph.ainvoke(Command(resume=decision))
      await_approval    returns the decision; approved args (or edited ones) become pending
      execute           draft_reply runs with approved=True and the approval id
      observe -> finish outcome = resolved
    RunRecord(status=completed, outcome) saved; ticket -> resolved
    MemoryWriter.remember(ticket, reply)   summary + embedding -> customer_memory
```

## State

`relay/agent/state.py` holds the graph state as plain JSON so it serialises cleanly into the
checkpointer:

| Field | Purpose |
|---|---|
| `messages` | Append-only conversation with the model (provider-neutral `Message` dicts) |
| `trajectory` | Append-only record of plans, tool calls, approvals, notes and the final outcome |
| `plan`, `replans` | Current plan (also in `messages` as an assistant turn) and how many times it changed |
| `pending` | The tool call waiting to run, plus `approved`/`approval_id` once a reviewer said yes |
| `tool_calls`, `consecutive_errors` | Counters behind the step limit and the replan/give-up rules |
| `approval` | The pending approval while the graph is interrupted |
| `sla_deadline`, `stop_reason` | SLA deadline for this ticket, and why `finish` was reached early |
| `outcome` | `resolved` or `escalated` with reason, reply or escalation id, and tool-call count |

## Persistence

| Store | Table / system | Written by |
|---|---|---|
| Tickets, customers | `tickets`, `customers` | API, runner (status changes) |
| Runs | `runs` | `AgentRunner` after every drive of the graph |
| Approvals | `pending_approvals` (unique `key`) | `request_approval` node, MCP server, reviewers |
| Escalations | `escalations` (reason `CHECK` constraint, trajectory `jsonb`) | `escalate_to_human`, policy hand-offs, resolvers |
| SLA policies | `sla_policies` | migration seed |
| Memory | `customer_memory` (`vector(256)`, HNSW cosine) | `MemoryWriter` |
| Graph state | LangGraph checkpoint tables | `AsyncPostgresSaver.setup()` / every super-step |
| Logs | OpenSearch index `app-logs` | your log pipeline (`scripts/seed_opensearch.py` for the demo) |

Migrations are plain SQL in `relay/db/migrations` applied by `relay/db/migrate.py`, which
records versions in `schema_migrations`. `002_memory.sql` is marked as needing the `vector`
extension; on a server without pgvector it is skipped with a warning and memory falls back to
the in-memory store.

## Resume semantics

Runs are invoked with `durability="sync"`, so a super-step is committed to the checkpointer
before the next one starts. Because `execute` contains exactly one tool call:

- a crash between steps resumes at the next node with nothing repeated;
- a crash inside `execute` repeats only that one call (at-least-once for the call in flight);
- a crash while waiting for approval loses nothing: the interrupt is in the checkpoint and the
  approval row is in Postgres, so a fresh process can apply the decision.

Write tools are the only calls with external effects, and they are reached only after an
approval whose key is `run_id:tool_call_id`.

## Tools and transports

`ToolRegistry` owns validation, the approval check, error shaping and the OpenTelemetry span.
`LocalToolClient` calls it in-process. `MCPToolClient` reaches the same registry through an MCP
session and passes the run id, trajectory and approval id in `_meta.relay`. On the server side
(`relay/mcp_server.py`), an unapproved write becomes a pending approval and returns
`{"status": "pending_approval", "approval_id": ...}`. A call with an approval id runs only if
that approval exists, is approved, and matches the tool and arguments exactly.

## Providers

All providers implement `ModelProvider.complete(ModelRequest) -> ModelResponse` with
provider-neutral messages, tool specs and tool calls. The Anthropic adapter merges consecutive
user turns, groups tool results into `tool_result` blocks, and replays each assistant turn's
raw content blocks (thinking included) unchanged; the system prompt is constant for a run, so
the history is append-only. The OpenAI adapter maps to
function calling, and unparseable argument JSON is passed through so validation reports it to
the model. The stub implements the same interface for four purposes: `plan`, `act`,
`summarize` and `judge`.

## Evaluation flow

```text
tasks.jsonl --load_tasks--> Task
Task --build_world--> InMemoryWorld (docs + fixtures, frozen clock) + Ticket
run_task: AgentRunner over the world, scripted reviewer approves writes,
          optional MCP transport -> TaskResult (trajectory, sent replies, escalations, latency)
score_task: task_success, trajectory_match, error_recovery, injection_resistance,
            injection_detected
build_report -> reports/eval_<ts>.json + .md (+ latest.*)
gate.py: minimums from gate.yaml, regression against baseline.json -> exit code
```

The eval runs the same graph, approval nodes, quoting and policy hand-offs as the API. Only
the storage and the reviewer are swapped out.
