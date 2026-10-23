"""Postgres connection and migration runner.

Migrations are plain SQL files in infra/migrations/, applied in name order,
each in its own transaction, recorded in mem.migrations. Re-running is a
no-op.
"""

from pathlib import Path

import psycopg

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "infra" / "migrations"


def connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(dsn, autocommit=False, connect_timeout=5)


def applied_migrations(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS mem")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS mem.migrations (
                migration_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                name         text NOT NULL UNIQUE,
                applied_at   timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute("SELECT name FROM mem.migrations")
        return {row[0] for row in cur.fetchall()}


def migrate(dsn: str) -> list[str]:
    """Apply all unapplied migrations. Returns the names it applied."""
    from agentmem.settings import get_settings  # local import avoids cycles

    applied = []
    dsn = dsn or get_settings().database_url
    files = sorted(p for p in MIGRATIONS_DIR.glob("*.sql"))
    with connect(dsn) as conn:
        done = applied_migrations(conn)
        for path in files:
            # "001_init.sql" -> "001_init"
            name = path.name.rsplit(".", 1)[0]
            if name in done:
                continue
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO mem.migrations (name) VALUES (%s)", (name,)
            )
            conn.commit()
            applied.append(name)
    return applied


def reset(dsn: str) -> None:
    """Drop the mem schema entirely (dev only)."""
    with connect(dsn) as conn:
        conn.execute("DROP SCHEMA IF EXISTS mem CASCADE")
        conn.commit()


if __name__ == "__main__":
    applied = migrate("")
    print("applied:", applied or "nothing to do")
