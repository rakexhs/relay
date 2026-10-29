"""Long-term customer memory: one summarized, embedded record per resolved ticket."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field

from relay.domain import utcnow
from relay.memory.embeddings import cosine


class MemoryRecord(BaseModel):
    customer_id: str
    ticket_id: str
    summary: str
    resolved_at: datetime = Field(default_factory=utcnow)
    embedding: list[float] = Field(default_factory=list, repr=False)


class MemoryHit(BaseModel):
    ticket_id: str
    summary: str
    resolved_at: datetime
    score: float


class MemoryStore(Protocol):
    def add(self, record: MemoryRecord) -> None: ...

    def search(
        self, customer_id: str, embedding: list[float] | None, top_k: int
    ) -> list[MemoryHit]: ...


class InMemoryMemoryStore:
    def __init__(self, records: list[MemoryRecord] | None = None) -> None:
        self._records: list[MemoryRecord] = list(records or [])

    def add(self, record: MemoryRecord) -> None:
        self._records = [
            r
            for r in self._records
            if not (r.customer_id == record.customer_id and r.ticket_id == record.ticket_id)
        ]
        self._records.append(record)

    def search(
        self, customer_id: str, embedding: list[float] | None, top_k: int
    ) -> list[MemoryHit]:
        mine = [r for r in self._records if r.customer_id == customer_id]
        if embedding is None:
            ranked = sorted(mine, key=lambda r: r.resolved_at, reverse=True)
            scored = [(r, 1.0) for r in ranked]
        else:
            scored = sorted(
                ((r, cosine(r.embedding, embedding)) for r in mine if r.embedding),
                key=lambda pair: pair[1],
                reverse=True,
            )
        return [
            MemoryHit(
                ticket_id=r.ticket_id,
                summary=r.summary,
                resolved_at=r.resolved_at,
                score=round(score, 4),
            )
            for r, score in scored[:top_k]
        ]
