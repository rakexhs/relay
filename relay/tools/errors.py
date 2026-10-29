"""Structured tool errors.

Every tool failure reaches the model as data with a stable shape:

    {"error": {"code": "invalid_query", "field": "query", "hint": "...", "retryable": false}}

A model can act on `field` and `hint` (fix the argument) or on `retryable` (try again later)
instead of repeating the same broken call until the step limit.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import ValidationError


class ToolErrorCode(StrEnum):
    INVALID_ARGUMENTS = "invalid_arguments"
    INVALID_QUERY = "invalid_query"
    NOT_FOUND = "not_found"
    UNKNOWN_TOOL = "unknown_tool"
    APPROVAL_REQUIRED = "approval_required"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    INTERNAL = "internal_error"


class ToolError(Exception):
    """Raised inside a tool handler; the registry turns it into an error payload."""

    def __init__(
        self,
        code: ToolErrorCode | str,
        hint: str,
        *,
        field: str | None = None,
        retryable: bool = False,
        **extra: Any,
    ) -> None:
        super().__init__(hint)
        self.code = ToolErrorCode(code)
        self.hint = hint
        self.field = field
        self.retryable = retryable
        self.extra = extra

    def payload(self) -> dict[str, Any]:
        return error_payload(
            self.code, self.hint, field=self.field, retryable=self.retryable, **self.extra
        )


def error_payload(
    code: ToolErrorCode | str,
    hint: str,
    *,
    field: str | None = None,
    retryable: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": str(ToolErrorCode(code)),
        "field": field,
        "hint": hint,
        "retryable": retryable,
    }
    body.update(extra)
    return {"error": body}


def is_error(result: Any) -> bool:
    return isinstance(result, dict) and isinstance(result.get("error"), dict)


def from_validation_error(exc: ValidationError) -> dict[str, Any]:
    """Turn a pydantic ValidationError into one structured error the model can act on."""
    issues = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(part) for part in err["loc"]) or None
        problem = err["msg"].removeprefix("Value error, ")
        issues.append({"field": loc, "problem": problem})
    first = issues[0] if issues else {"field": None, "problem": "invalid arguments"}
    hint = first["problem"]
    if first["field"]:
        hint = f"{first['field']}: {hint}"
    return error_payload(
        ToolErrorCode.INVALID_ARGUMENTS,
        hint,
        field=first["field"],
        issues=issues,
    )
