"""Supersession guard: stale values stated as current are detected."""

import random

import psycopg
import pytest

from agentmem.agent import _stale_hits
from agentmem.db import migrate
from agentmem.settings import get_settings
from agentmem.units import entity


def test_stale_hits_detects_old_without_current():
    pairs = [{"name": "maya", "attr": "income_monthly",
              "old": 7900, "current": 8750}]
    assert _stale_hits("your income is $7,900", pairs) == pairs
    assert _stale_hits("your income is $8,750", pairs) == []
    # faithful recall mentions the old value but also states the current one
    assert _stale_hits("you said 7,900 earlier; it is now 8,750", pairs) == []


@pytest.fixture()
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    for table in ("mem.fact", "mem.entity"):
        c.execute(f"DELETE FROM {table} WHERE user_id LIKE 'sg-%'")
    c.close()


def test_superseded_pairs_returns_old_and_current(conn):
    u = f"sg-{random.randint(10**8, 10**9 - 1)}"
    entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                      name=u, attr="income_monthly", value=7900)
    entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                      name=u, attr="income_monthly", value=8750)
    pairs = entity.superseded_pairs(conn, user_id=u)
    assert pairs == [{"name": u, "attr": "income_monthly",
                      "old": 7900, "current": 8750}]
