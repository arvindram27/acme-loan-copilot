"""Unit 6 contract tests: working memory + semantic cache — no model calls.

Summarization is exercised through an injected deterministic summarizer, so
the mechanics (folding boundary, versioning, id collection) are tested without
the LLM. The cache is tested with hand-built one-hot vectors.
"""

import pytest
import psycopg

from agentmem.db import migrate
from agentmem.settings import get_settings
from agentmem.units import cache, conversation, entity, working

DIMS = 1536


def _vec(slot: int) -> list[float]:
    v = [0.0] * DIMS
    v[slot % DIMS] = 1.0
    return v


@pytest.fixture(scope="module")
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    for table in ("mem.cache", "mem.summary", "mem.turn", "mem.fact",
                  "mem.entity"):
        c.execute(f"DELETE FROM {table} WHERE user_id LIKE 'm5-%'")
    c.close()


def _user():
    import random
    return f"m5-{random.randint(10**8, 10**9 - 1)}"


# --- semantic answer cache --------------------------------------------------

def test_fact_epoch_starts_at_zero(conn):
    assert cache.fact_epoch(conn, _user()) == 0


def test_store_then_lookup_hit(conn):
    u = _user()
    cache.store(conn, user_id=u, thread_id="t", question="what is my income?",
                query_embedding=_vec(0), answer="8,750")
    hit = cache.lookup(conn, user_id=u, thread_id="t",
                       query_embedding=_vec(0))
    assert hit is not None
    assert hit["answer"] == "8,750"
    assert hit["similarity"] == pytest.approx(1.0)


def test_lookup_miss_across_threads(conn):
    u = _user()
    cache.store(conn, user_id=u, thread_id="a", question="q",
                query_embedding=_vec(0), answer="x")
    assert cache.lookup(conn, user_id=u, thread_id="b",
                        query_embedding=_vec(0)) is None


def test_lookup_miss_below_similarity_threshold(conn):
    u = _user()
    cache.store(conn, user_id=u, thread_id="t", question="income",
                query_embedding=_vec(0), answer="x")
    # orthogonal vector -> similarity ~0, far below 0.92
    assert cache.lookup(conn, user_id=u, thread_id="t",
                        query_embedding=_vec(1)) is None


def test_fact_write_invalidates_cache(conn):
    """The epoch guard: a new fact bumps the user's epoch, so a cached answer
    stored before the write is no longer reachable."""
    u = _user()
    cache.store(conn, user_id=u, thread_id="t", question="income",
                query_embedding=_vec(0), answer="8,750")
    entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                      name=u, attr="income_monthly", value=9000)
    assert cache.fact_epoch(conn, u) > 0
    assert cache.lookup(conn, user_id=u, thread_id="t",
                        query_embedding=_vec(0)) is None


def test_expired_entry_misses(conn):
    u = _user()
    cache.store(conn, user_id=u, thread_id="t", question="income",
                query_embedding=_vec(0), answer="x")
    conn.execute(
        "UPDATE mem.cache SET expires_at = now() - interval '1 minute' "
        "WHERE user_id = %s", (u,),
    )
    assert cache.lookup(conn, user_id=u, thread_id="t",
                        query_embedding=_vec(0)) is None


def test_is_question_gate():
    assert cache.is_question("what is my monthly income?")
    assert cache.is_question("could you tell me my income per month?")
    assert cache.is_question("remind me of my rate")
    assert not cache.is_question("I make 9,000 a month now")
    assert not cache.is_question("my car loan cleared last week")


# --- working memory summaries -----------------------------------------------

def test_no_summary_within_window(conn):
    u = _user()
    for i in range(4):
        conversation.append_turn(conn, user_id=u, thread_id="w1",
                                 role="user", content=f"turn {i}")
    out = working.summarize_older(conn, user_id=u, thread_id="w1",
                                  summarizer=lambda m: m, keep_k=4)
    assert out is None


def test_summary_after_window(conn):
    u = _user()
    for i in range(6):
        conversation.append_turn(conn, user_id=u, thread_id="w2",
                                 role="user", content=f"turn {i}")
    out = working.summarize_older(conn, user_id=u, thread_id="w2",
                                  summarizer=lambda m: f"SUM[{m}]", keep_k=4)
    assert out is not None
    # 6 turns, keep 4 -> the 2 oldest are folded
    assert out["covers_through_seq"] == 2
    assert len(out["source_turn_ids"]) == 2
    assert "turn 0" in out["text"] and "turn 1" in out["text"]
    assert "turn 5" not in out["text"]  # newest stays verbatim, not folded
    active = working.read_active(conn, u, "w2")
    assert active["version"] == 1
    assert active["covers_through_seq"] == 2


def test_summary_increments_version_and_extends(conn):
    u = _user()
    for i in range(6):
        conversation.append_turn(conn, user_id=u, thread_id="w3",
                                 role="user", content=f"turn {i}")
    first = working.summarize_older(conn, user_id=u, thread_id="w3",
                                    summarizer=lambda m: f"SUM[{m}]", keep_k=4)
    # two more turns -> boundary advances by 2, version bumps, ids accumulate
    conversation.append_turn(conn, user_id=u, thread_id="w3",
                             role="user", content="turn 6")
    conversation.append_turn(conn, user_id=u, thread_id="w3",
                             role="user", content="turn 7")
    second = working.summarize_older(conn, user_id=u, thread_id="w3",
                                     summarizer=lambda m: f"SUM[{m}]", keep_k=4)
    assert second["version"] == first["version"] + 1
    assert second["covers_through_seq"] == first["covers_through_seq"] + 2
    assert len(second["source_turn_ids"]) == len(first["source_turn_ids"]) + 2
    # the new summary folds the prior summary text plus the newly-old turns
    assert "SUM[" in second["text"]
    active = working.read_active(conn, u, "w3")
    assert active["version"] == second["version"]
