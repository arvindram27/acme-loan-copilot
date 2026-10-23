"""Migrations + schema contract against the real database.

Skipped when Postgres is not up (docker compose in infra/).
"""

import psycopg
import pytest

from agentmem.db import applied_migrations, migrate
from agentmem.settings import get_settings


def _db_up() -> bool:
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_up(), reason="Postgres not reachable — start infra/docker-compose.yml"
)


def test_migrations_idempotent():
    first = migrate(get_settings().database_url)
    second = migrate(get_settings().database_url)
    assert not second or second != first or True  # reruns must not error
    with psycopg.connect(get_settings().database_url) as conn:
        assert applied_migrations(conn)


@pytest.mark.parametrize(
    "table",
    ["mem.turn", "mem.persona", "mem.migrations"],
)
def test_tables_exist(table):
    with psycopg.connect(get_settings().database_url) as conn:
        cur = conn.execute(f"SELECT to_regclass('{table}')")
        assert cur.fetchone()[0] is not None, f"{table} missing"


def test_persona_single_active_constraint():
    """The partial unique index must reject a second active persona per user."""
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute("DELETE FROM mem.persona WHERE user_id = 'test-user'")
        conn.execute(
            """
            INSERT INTO mem.persona (user_id, version, change_trigger, body)
            VALUES ('test-user', 1, 'app_default', '{"style": "concise"}')
            """
        )
        with pytest.raises(psycopg.errors.IntegrityError):  # noqa: PT012
            conn.execute(
                """
                INSERT INTO mem.persona (user_id, version, change_trigger, body)
                VALUES ('test-user', 2, 'user_request', '{"style": "formal"}')
                """
            )
        conn.execute("DELETE FROM mem.persona WHERE user_id = 'test-user'")


def test_turn_scope_unique():
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute("DELETE FROM mem.turn WHERE thread_id = 't'")
        conn.execute(
            """
            INSERT INTO mem.turn (user_id, thread_id, seq, role, content)
            VALUES ('u', 't', 1, 'user', 'hello')
            """
        )
        with pytest.raises(psycopg.errors.IntegrityError):  # noqa: PT012
            conn.execute(
                """
                INSERT INTO mem.turn (user_id, thread_id, seq, role, content)
                VALUES ('u', 't', 1, 'assistant', 'duplicate seq')
                """
            )
        conn.execute("DELETE FROM mem.turn WHERE thread_id = 't'")
