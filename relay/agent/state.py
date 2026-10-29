"""Graph state. Everything here is plain JSON so it checkpoints cleanly."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Route = Literal["act", "replan", "finish"]


class AgentState(TypedDict, total=False):
    run_id: str
    ticket: dict[str, Any]
    # Conversation with the model, as relay.providers.Message dicts. Append-only.
    messages: Annotated[list[dict[str, Any]], operator.add]
    # One record per plan, tool call, approval decision and final outcome. Append-only.
    trajectory: Annotated[list[dict[str, Any]], operator.add]
    plan: list[str]
    replans: int
    # The tool call the model asked for and that has not run yet.
    pending: dict[str, Any] | None
    tool_calls: int
    consecutive_errors: int
    route: Route
    # Set when a write tool is waiting on a reviewer.
    approval: dict[str, Any] | None
    # ISO timestamp of the SLA resolution deadline; past it the run escalates with sla_breach.
    sla_deadline: str | None
    stop_reason: str | None
    final_text: str
    outcome: dict[str, Any] | None
