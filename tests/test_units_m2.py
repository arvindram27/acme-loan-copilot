"""Unit 3 contract tests: entity memory with supersession, no model calls."""

import pytest
import psycopg

from agentmem.db import migrate
from agentmem.settings import get_settings
from agentmem.units import entity


@pytest.fixture(scope="module")
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    c.execute("DELETE FROM mem.fact WHERE user_id LIKE 'm2-%'")
    c.execute("DELETE FROM mem.entity WHERE user_id LIKE 'm2-%'")
    c.close()


def test_write_and_read(conn):
    out = entity.write_fact(conn, user_id="m2-u", thread_id="m2-t1",
                            kind="applicant", name="Jane Doe",
                            attr="income_annual", value=95000)
    assert out["changed"] and out["supersedes_id"] is None
    facts = entity.current_facts(conn, entity.resolve(
        conn, user_id="m2-u", kind="applicant", name="Jane Doe")["entity_id"])
    assert facts == {"income_annual": 95000}


def test_supersession_excludes_stale(conn):
    """The demo moment: income corrected, the old value must not resolve."""
    e = entity.resolve(conn, user_id="m2-u", kind="applicant",
                       name="Jane Doe")["entity_id"]
    entity.write_fact(conn, user_id="m2-u", thread_id="m2-t1", kind="applicant",
                      name="Jane Doe", attr="income_annual", value=105000,
                      source_ref="turn:12")
    facts = entity.current_facts(conn, e)
    assert facts["income_annual"] == 105000
    history = entity.fact_history(conn, e, "income_annual")
    assert [h["value"] for h in history] == [95000, 105000]
    assert [h["status"] for h in history] == ["superseded", "active"]
    assert history[1]["supersedes_id"] == history[0]["fact_id"]


def test_same_value_write_is_noop(conn):
    e = entity.resolve(conn, user_id="m2-u", kind="applicant",
                       name="Jane Doe")["entity_id"]
    before = entity.fact_history(conn, e, "income_annual")
    out = entity.write_fact(conn, user_id="m2-u", thread_id="m2-t2",
                            kind="applicant", name="Jane Doe",
                            attr="income_annual", value=105000)
    assert not out["changed"]
    assert entity.fact_history(conn, e, "income_annual") == before


def test_learned_in_one_thread_read_in_another(conn):
    """A fact learned in t1 is visible from t2 — entity memory is cross-thread."""
    facts = entity.current_facts(conn, entity.resolve(
        conn, user_id="m2-u", kind="applicant", name="Jane Doe")["entity_id"])
    assert facts["income_annual"] == 105000  # learned in thread m2-t1


def test_one_active_fact_invariant(conn):
    """The partial unique index rejects a second active fact for the same key."""
    data = (conn, "m2-cap", "m2-t", "property", "48 Elm St", "units", 2)
    entity.write_fact(conn, user_id=data[1], thread_id=data[2], kind=data[3],
                      name=data[4], attr=data[5], value=data[6])
    with pytest.raises(psycopg.errors.IntegrityError):  # noqa: PT012
        conn.execute(
            """
            INSERT INTO mem.fact (entity_id, user_id, thread_id, attr, value)
            VALUES (
              (SELECT entity_id FROM mem.entity
                WHERE user_id = 'm2-cap' AND kind = 'property' AND name = '48 Elm St'),
              'm2-cap', 'm2-t', 'units', '3')
            """
        )


def test_unknown_entity_is_none(conn):
    assert entity.resolve(conn, user_id="m2-u", kind="applicant",
                          name="Nobody McMissing") is None


def test_attr_allowlist(conn):
    with pytest.raises(ValueError):
        entity.write_fact(conn, user_id="m2-u", thread_id="m2-t",
                          kind="applicant", name="Jane Doe",
                          attr="favorite_color", value="teal")


def test_entity_kinds_are_scoped(conn):
    """Same name, different kinds -> different entities, independent facts."""
    entity.write_fact(conn, user_id="m2-u2", thread_id="m2-t",
                      kind="property", name="48 Elm St", attr="units", value=2)
    entity.write_fact(conn, user_id="m2-u2", thread_id="m2-t",
                      kind="loan", name="48 Elm St", attr="loan_amount", value=320000)
    prop = entity.resolve(conn, user_id="m2-u2", kind="property", name="48 Elm St")
    loan = entity.resolve(conn, user_id="m2-u2", kind="loan", name="48 Elm St")
    assert prop["facts"].keys() == {"units"}
    assert loan["facts"].keys() == {"loan_amount"}


pytestmark = pytest.mark.db
