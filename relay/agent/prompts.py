"""Prompts and the rendering of tickets and tool results into model messages."""

from __future__ import annotations

import json
from typing import Any

from relay.safety import DATA_RULE, InjectionReport, quote_untrusted, scan

SYSTEM_PROMPT = """You are Relay, a support engineer working one customer ticket at a time.

You have six tools: search_docs, lookup_ticket, query_logs, draft_reply, escalate_to_human and
recall_customer_history. Call one tool at a time and read each result before the next call.

How to work a ticket:
- Start with lookup_ticket, then gather evidence (docs, logs) that fits the customer's problem.
- Answer only from what the tools returned. Do not invent features, settings or timelines.
- When the answer is clear, call draft_reply with send=true. A human approves it before it goes out.
- Escalate with escalate_to_human and the matching reason when you cannot finish:
  needs_refund (any refund or credit request: you cannot issue them), customer_request (they
  asked for a person), tool_failure (a tool keeps failing), low_confidence (no evidence for an
  answer). Include what you tried in the summary.
- If a tool returns {"error": {...}}, read code, field and hint, fix that argument and retry.
  Do not repeat a call that already failed the same way.
- When the reply is sent or the ticket is escalated, stop calling tools and say you are done.

"""
SYSTEM_PROMPT += DATA_RULE

PLAN_INSTRUCTION = (
    "Before acting, write a short numbered plan (3-6 steps) for this ticket. "
    "Name the tools you expect to use. Reply with the plan only."
)

REPLAN_INSTRUCTION = (
    "Your recent tool calls failed. Look at the errors above and write a revised numbered plan "
    "(3-6 steps). Reply with the plan only."
)


PROCEED_INSTRUCTION = "Carry out the plan, one tool call at a time."


def render_ticket(
    ticket: dict[str, Any], memories: list[dict[str, Any]], *, quote: bool = True
) -> tuple[str, list[InjectionReport]]:
    """The opening user message. The ticket body and memories are customer-controlled text."""
    reports = []
    parts = ["New ticket (the subject and body were written by the customer):"]
    if quote:
        block, report = quote_untrusted("ticket", ticket)
        parts.append(block)
        reports.append(report)
    else:
        parts.append(json.dumps(ticket, indent=2))
        reports.append(scan(ticket))
    if memories:
        parts.append("Summaries of this customer's past resolved tickets:")
        payload = {"memories": memories}
        if quote:
            block, report = quote_untrusted("recall_customer_history", payload)
            parts.append(block)
            reports.append(report)
        else:
            parts.append(json.dumps(payload, indent=2))
            reports.append(scan(payload))
    return "\n\n".join(parts), reports


def render_observation(
    tool: str, result: dict[str, Any], *, quote: bool = True
) -> tuple[str, InjectionReport]:
    if quote:
        block, report = quote_untrusted(tool, result)
        return f"Result of {tool}:\n{block}", report
    return f"Result of {tool}:\n{json.dumps(result, default=str)}", scan(result)


def parse_plan(text: str) -> list[str]:
    steps = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line:
            continue
        head, sep, rest = line.partition(".")
        if sep and head.isdigit():
            line = rest.strip()
        steps.append(line)
    return steps[:8]
