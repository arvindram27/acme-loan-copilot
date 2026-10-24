"""Unit-of-work policy: caller-owned transactions and nested savepoints.

The bug this guards against: a unit calling conn.commit() would commit an
outer transaction early, so a multi-write turn could be half-persisted.
"""

import random
from pathlib import Path

import psycopg
import pytest

from agentmem.db import migrate
from agentmem.settings import get_settings
from agentmem.units import entity, persona


@pytest.fixture(scope="module")
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    for table in ("mem.fact", "mem.entity", "mem.persona"):
        c.execute(f"DELETE FROM {table} WHERE user_id LIKE 'tx-%'")
    c.close()


def _user() -> str:
    return f"tx-{random.randint(10**8, 10**9 - 1)}"


def _fact_count(conn, user: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM mem.fact WHERE user_id = %s", (user,)
    ).fetchone()[0]


def test_outer_rollback_discards_unit_write(conn):
    u = _user()
    before = _fact_count(conn, u)
    with pytest.raises(RuntimeError):
        with conn.transaction():
            entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                              name=u, attr="income_monthly", value=1000)
            raise RuntimeError("boom after the write")
    assert _fact_count(conn, u) == before, "the write must not survive rollback"


def test_outer_commit_keeps_all_nested_writes(conn):
    u = _user()
    with conn.transaction():
        entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                          name=u, attr="income_monthly", value=1000)
        entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                          name=u, attr="monthly_debts", value=500)
    facts = entity.resolve(conn, user_id=u, kind="applicant", name=u)["facts"]
    assert facts == {"income_monthly": 1000, "monthly_debts": 500}


def test_persona_update_participates_in_outer_transaction(conn):
    u = _user()
    persona.ensure_default(conn, u)
    v1 = persona.read_active(conn, u)["version"]
    with pytest.raises(RuntimeError):
        with conn.transaction():
            persona.update(conn, u, {"style": "x", "rules": []})
            raise RuntimeError("boom")
    assert persona.read_active(conn, u)["version"] == v1, \
        "the persona update must roll back with the outer transaction"


def test_no_unit_calls_commit_directly():
    root = Path(__file__).resolve().parents[1] / "src" / "agentmem" / "units"
    offenders = [
        p.name for p in root.glob("*.py")
        if "conn.commit()" in p.read_text(encoding="utf-8")
        or "conn.rollback()" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"units must not own the commit boundary: {offenders}"
