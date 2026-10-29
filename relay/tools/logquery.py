"""A small, strict log query language and its translation to OpenSearch DSL.

Grammar:  clause ("AND" clause)*      clause := ["NOT"] field ":" value
Values may be bare words or double-quoted strings. OR is rejected on purpose: one query
should answer one question, and the error tells the model to split it.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from relay.tools.errors import ToolError, ToolErrorCode

FIELDS = ("service", "level", "status_code", "customer_id", "request_id", "message")
LEVELS = ("debug", "info", "warn", "error")


@dataclass(frozen=True)
class Clause:
    field: str
    value: str
    negate: bool = False

    def matches(self, record: dict[str, Any]) -> bool:
        actual = record.get(self.field)
        if self.field == "message":
            hit = self.value.lower() in str(actual or "").lower()
        else:
            hit = str(actual).lower() == self.value.lower()
        return hit != self.negate


@dataclass(frozen=True)
class LogQuery:
    clauses: tuple[Clause, ...]

    def matches(self, record: dict[str, Any]) -> bool:
        return all(clause.matches(record) for clause in self.clauses)


def _invalid(hint: str, **extra: Any) -> ToolError:
    return ToolError(ToolErrorCode.INVALID_QUERY, hint, field="query", **extra)


def parse_query(text: str) -> LogQuery:
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        raise _invalid(
            f"Could not tokenize the query ({exc}). Check for an unclosed quote."
        ) from exc
    if not tokens:
        raise _invalid("The query is empty. Example: service:api AND level:error")

    clauses: list[Clause] = []
    expect_clause = True
    negate = False
    for token in tokens:
        upper = token.upper()
        if upper == "OR":
            raise _invalid("OR is not supported. Run one query per alternative.")
        if upper == "AND":
            if expect_clause:
                raise _invalid("AND must sit between two field:value clauses.")
            expect_clause = True
            continue
        if upper == "NOT":
            negate = True
            continue
        if not expect_clause:
            raise _invalid(f"Missing AND before '{token}'. Join clauses with AND.")
        field, sep, value = token.partition(":")
        if not sep or not field or not value:
            raise _invalid(
                f"'{token}' is not a field:value clause. Example: level:error",
                allowed=list(FIELDS),
            )
        field = field.lower()
        if field not in FIELDS:
            raise _invalid(
                f"Unknown field '{field}'. Allowed fields: {', '.join(FIELDS)}.",
                allowed=list(FIELDS),
            )
        if field == "status_code" and not re.fullmatch(r"\d{3}", value):
            raise _invalid(f"status_code must be a 3-digit HTTP code, got '{value}'.")
        if field == "level" and value.lower() not in LEVELS:
            raise _invalid(f"level must be one of {', '.join(LEVELS)}, got '{value}'.")
        clauses.append(Clause(field, value, negate))
        negate = False
        expect_clause = False
    if expect_clause:
        raise _invalid("The query ends with AND or NOT. Remove it or add a clause.")
    return LogQuery(tuple(clauses))


def parse_since(value: str) -> timedelta:
    amount, unit = int(value[:-1]), value[-1]
    return {
        "m": timedelta(minutes=amount),
        "h": timedelta(hours=amount),
        "d": timedelta(days=amount),
    }[unit]


def to_opensearch(query: LogQuery, since: str, limit: int) -> dict[str, Any]:
    must: list[dict[str, Any]] = [{"range": {"@timestamp": {"gte": f"now-{since}"}}}]
    must_not: list[dict[str, Any]] = []
    for clause in query.clauses:
        if clause.field == "message":
            cond: dict[str, Any] = {"match_phrase": {"message": clause.value}}
        elif clause.field == "status_code":
            cond = {"term": {"status_code": int(clause.value)}}
        else:
            cond = {"term": {f"{clause.field}.keyword": clause.value}}
        (must_not if clause.negate else must).append(cond)
    return {
        "size": limit,
        "sort": [{"@timestamp": {"order": "desc"}}],
        "query": {"bool": {"must": must, "must_not": must_not}},
    }
