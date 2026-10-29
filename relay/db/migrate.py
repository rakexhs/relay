"""Apply the SQL migrations in relay/db/migrations in order. Safe to run repeatedly.

python -m relay.db.migrate            # uses RELAY_DATABASE_URL
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import psycopg

from relay.config import get_settings

log = logging.getLogger(__name__)

MIGRATIONS = Path(__file__).resolve().parent / "migrations"
NEEDS_VECTOR = "-- requires: vector"


def vector_available(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'").fetchone()
    return row is not None


def migrate(database_url: str) -> list[str]:
    """Apply pending migrations and return the versions applied in this call."""
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        has_vector = vector_available(conn)
        for path in sorted(MIGRATIONS.glob("*.sql")):
            version = path.stem
            if version in done:
                continue
            sql = path.read_text(encoding="utf-8")
            if sql.startswith(NEEDS_VECTOR) and not has_vector:
                log.warning("skipping %s: the pgvector extension is not installed", version)
                continue
            with conn.transaction():
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
    return applied


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    url = get_settings().database_url
    if not url:
        print("RELAY_DATABASE_URL is not set", file=sys.stderr)
        return 1
    applied = migrate(url)
    print(f"applied: {', '.join(applied) if applied else 'nothing new'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
