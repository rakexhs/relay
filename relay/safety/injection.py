"""A small heuristic detector for instruction-like text inside retrieved content.

It is a tripwire, not the defense. The defense is that retrieved text is always quoted as data
(see quoting.py) and the system prompt says data is never instructions. The detector adds a
visible flag to the block, records a safety event in the trajectory, and gives the eval a way
to check that planted instructions were noticed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

TOOL_NAMES = (
    r"(search_docs|lookup_ticket|query_logs|draft_reply|escalate_to_human|recall_customer_history)"
)

RULES: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "override_instructions",
        1.0,
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,30}\b(previous|prior|above|earlier|all|your|system)\b"
            r"[^.\n]{0,20}\b(instructions?|rules|prompt|guidelines|messages)\b",
            re.I,
        ),
    ),
    (
        "role_hijack",
        1.0,
        re.compile(
            r"(^|\n)\s*(system|assistant|developer)\s*:|\byou are now\b|\bnew instructions\b", re.I
        ),
    ),
    (
        "tool_directive",
        1.0,
        re.compile(rf"\b(call|use|invoke|run|trigger)\s+(the\s+)?{TOOL_NAMES}\b", re.I),
    ),
    (
        "reply_directive",
        0.6,
        re.compile(
            r"\b(reply|respond|tell|send)\b[^.\n]{0,40}\b(saying|with the text|with this|exactly)\b",
            re.I,
        ),
    ),
    (
        "refund_directive",
        0.8,
        re.compile(r"\b(issue|approve|grant|process)\s+(a\s+)?(full\s+)?(refund|credit)\b", re.I),
    ),
    (
        "secrecy",
        0.6,
        re.compile(
            r"\bdo not (tell|inform|mention|reveal)\b[^.\n]{0,30}\b(user|customer|anyone)\b", re.I
        ),
    ),
    (
        "exfiltration",
        1.0,
        re.compile(
            r"\b(send|reveal|include|paste|share)\b[^.\n]{0,40}\b(api[ _-]?keys?|passwords?|secrets?|tokens?|credentials)\b",
            re.I,
        ),
    ),
    (
        "hidden_markup",
        0.4,
        re.compile(r"<!--|\[\[\s*instructions?|BEGIN (SYSTEM|INSTRUCTIONS)", re.I),
    ),
)

FLAG_THRESHOLD = 1.0


@dataclass(frozen=True)
class Finding:
    rule: str
    weight: float
    excerpt: str


@dataclass
class InjectionReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def score(self) -> float:
        return round(sum(f.weight for f in self.findings), 2)

    @property
    def flagged(self) -> bool:
        return self.score >= FLAG_THRESHOLD

    @property
    def rules(self) -> list[str]:
        return sorted({f.rule for f in self.findings})

    def as_dict(self) -> dict[str, Any]:
        return {
            "flagged": self.flagged,
            "score": self.score,
            "rules": self.rules,
            "excerpts": [f.excerpt for f in self.findings[:3]],
        }


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)


def scan(value: Any) -> InjectionReport:
    """Scan a string or any JSON-like structure for instruction-like content."""
    report = InjectionReport()
    seen: set[str] = set()
    for text in _strings(value):
        for rule, weight, pattern in RULES:
            if rule in seen:
                continue
            match = pattern.search(text)
            if match:
                seen.add(rule)
                start = max(match.start() - 30, 0)
                excerpt = text[start : match.end() + 60].replace("\n", " ").strip()
                report.findings.append(Finding(rule, weight, excerpt))
    return report
