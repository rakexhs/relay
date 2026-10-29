"""Customer memory in Postgres with pgvector (migration 002_memory.sql).

Vectors are sent as pgvector text literals and cast in SQL, so the pgvector Python package is
not required. Search is per customer, ranked by cosine distance on an HNSW index.
"""

from __future__ import annotations

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from relay.memory.store import MemoryHit, MemoryRecord


def to_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


class PgVectorMemoryStore:
    def __init__(self, pool: ConnectionPool | str) -> None:
        if isinstance(pool, str):
            pool = ConnectionPool(
                pool, min_size=1, max_size=3, kwargs={"row_factory": dict_row, "autocommit": True}
            )
        self.pool = pool

    def add(self, record: MemoryRecord) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO customer_memory (customer_id, ticket_id, summary, resolved_at, embedding) "
                "VALUES (%s, %s, %s, %s, %s::vector) ON CONFLICT (customer_id, ticket_id) DO UPDATE "
                "SET summary = EXCLUDED.summary, resolved_at = EXCLUDED.resolved_at, "
                "embedding = EXCLUDED.embedding",
                (
                    record.customer_id,
                    record.ticket_id,
                    record.summary,
                    record.resolved_at,
                    to_literal(record.embedding),
                ),
            )

    def search(
        self, customer_id: str, embedding: list[float] | None, top_k: int
    ) -> list[MemoryHit]:
        with self.pool.connection() as conn:
            if embedding is None:
                rows = conn.execute(
                    "SELECT ticket_id, summary, resolved_at, 1.0 AS score FROM customer_memory "
                    "WHERE customer_id = %s ORDER BY resolved_at DESC LIMIT %s",
                    (customer_id, top_k),
                ).fetchall()
            else:
                literal = to_literal(embedding)
                rows = conn.execute(
                    "SELECT ticket_id, summary, resolved_at, "
                    "1 - (embedding <=> %s::vector) AS score FROM customer_memory "
                    "WHERE customer_id = %s ORDER BY embedding <=> %s::vector LIMIT %s",
                    (literal, customer_id, literal, top_k),
                ).fetchall()
        return [
            MemoryHit(
                ticket_id=r["ticket_id"],
                summary=r["summary"],
                resolved_at=r["resolved_at"],
                score=round(float(r["score"]), 4),
            )
            for r in rows
        ]
