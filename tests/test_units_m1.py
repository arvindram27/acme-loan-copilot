"""Unit 1 + 2 contract tests: conversation and persona against real Postgres.

No model calls: embeddings are injected as deterministic float vectors.
"""

import pytest
import psycopg

from agentmem.db import migrate
from agentmem.settings import get_settings
from agentmem.units import conversation, persona


@pytest.fixture(scope="module")
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    c.execute("DELETE FROM mem.turn WHERE thread_id LIKE 'm1-%'")
    c.execute("DELETE FROM mem.persona WHERE user_id IN ('m1-user')")
    c.close()


DIMS = 1536  # must match the schema's pinned vector dimension


def _fake_vec(slot: int) -> list[float]:
    """Deterministic 1536-d one-hot±offset vector, orthogonal-ish per slot."""
    v = [0.0] * DIMS
    v[slot % DIMS] = 1.0
    return v


# --- conversation -----------------------------------------------------------

def test_append_turn_seq_autonumber(conn):
    a = conversation.append_turn(conn, user_id="m1-user", thread_id="m1-1",
                                 role="user", content="What is the max LTV?")
    b = conversation.append_turn(conn, user_id="m1-user", thread_id="m1-1",
                                 role="assistant", content="95% for conforming.")
    assert a["seq"] == b["seq"] - 1


def test_scope_isolation(conn):
    """Turns in one thread never appear in another thread's reads."""
    conversation.append_turn(conn, user_id="m1-user", thread_id="m1-1",
                             role="user", content="thread one turn")
    conversation.append_turn(conn, user_id="m1-user", thread_id="m1-2",
                             role="user", content="thread two turn")
    recent = conversation.recent_turns(conn, "m1-2", k=10)
    assert [t["content"] for t in recent] == ["thread two turn"]


def test_recent_turns_oldest_first(conn):
    for i in range(5):
        conversation.append_turn(conn, user_id="m1-user", thread_id="m1-3",
                                 role="user", content=f"turn {i}")
    recent = conversation.recent_turns(conn, "m1-3", k=3)
    assert [t["content"] for t in recent] == ["turn 2", "turn 3", "turn 4"]


def test_semantic_recall(conn):
    # seq 1 "embeds" as a one-hot at slot 0, seq 7 at slot 5
    conversation.append_turn(conn, user_id="m1-user", thread_id="m1-4",
                             role="user", content="borrower income details",
                             embedding=_fake_vec(0))
    conversation.append_turn(conn, user_id="m1-user", thread_id="m1-4",
                             role="user", content="property type talk",
                             embedding=_fake_vec(5))
    hits = conversation.recall(conn, user_id="m1-user", thread_id="m1-4",
                               query_embedding=_fake_vec(0), k=2)
    assert hits[0]["content"] == "borrower income details"
    assert hits[0]["similarity"] == pytest.approx(1.0)
    # before_seq keeps recall behind the recency window: with the window
    # starting at seq 1 no older turns exist; starting at 2, only seq 1 is a
    # candidate, so the window's own turns are never duplicated in context.
    assert conversation.recall(conn, user_id="m1-user", thread_id="m1-4",
                               query_embedding=_fake_vec(0), k=2,
                               before_seq=1) == []
    only_income = conversation.recall(conn, user_id="m1-user", thread_id="m1-4",
                                      query_embedding=_fake_vec(0), k=2,
                                      before_seq=2)
    assert [h["content"] for h in only_income] == ["borrower income details"]


def test_embedding_null_still_queryable(conn):
    """Turns without embeddings are visible to recency but never to recall."""
    conversation.append_turn(conn, user_id="m1-user", thread_id="m1-5",
                             role="user", content="no vector here")
    hits = conversation.recall(conn, user_id="m1-user", thread_id="m1-5",
                               query_embedding=_fake_vec(3), k=3)
    assert hits == []
    assert len(conversation.recent_turns(conn, "m1-5")) == 1


# --- persona ----------------------------------------------------------------

def test_ensure_default_idempotent(conn):
    first = persona.ensure_default(conn, "m1-user")
    again = persona.ensure_default(conn, "m1-user")
    assert first["version"] == again["version"] == 1
    assert first["change_trigger"] == "app_default"


def test_update_supersedes(conn):
    v1 = persona.read_active(conn, "m1-user")
    v2 = persona.update(conn, "m1-user", {"style": "extra formal", "rules": []},
                        change_trigger="user_request")
    assert v1["version"] == 1 and v2["version"] == 2
    assert persona.read_version(conn, "m1-user", 1)["status"] == "superseded"
    assert persona.read_active(conn, "m1-user")["version"] == 2


def test_old_version_still_readable(conn):
    """History is kept: superseded versions remain queryable by version."""
    v1 = persona.read_version(conn, "m1-user", 1)
    assert v1 is not None and v1["change_trigger"] == "app_default"


def test_format_persona(conn):
    text = persona.format_persona(persona.read_active(conn, "m1-user"))
    assert "Persona (version" in text


pytestmark = pytest.mark.db
