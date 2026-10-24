"""LangGraph backend contract tests, offline: no model or API calls.

The store/checkpointer mechanics (fact supersession, cross-thread visibility,
scope isolation, turn recall, persona) are exercised against the real
agentmem_lg database with a deterministic fake embedder. A one-turn graph
round-trip with a fake chat locks the config/thread_id wiring.
"""

import pytest
import psycopg

from langchain_core.messages import AIMessage, HumanMessage

from agentmem.backends.langgraph import (
    LangGraphAgent, build_checkpointer, build_store,
)
from agentmem.db import migrate
from agentmem.settings import get_settings

DIMS = 1536


class FakeEmbedder:
    def embed(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIMS
            v[hash(t) % DIMS] = 1.0
            out.append(v)
        return out


class FakeChat:
    """Deterministic: no facts, no tools, a canned final answer."""

    def complete(self, messages, **kw):
        system = messages[0]["content"] if messages else ""
        class R:
            total_tokens = 7
            prompt_tokens = 4
        r = R()
        if "Extract factual statements" in system:
            r.text = '{"facts": []}'
            r.model = "fake"
        elif "tool-planning" in system or "decide the next step" in system:
            r.text = '{"action": "answer"}'
            r.model = "fake"
        else:
            r.text = "8,750"
            r.model = "fake"
        return r


@pytest.fixture(scope="module")
def lg():
    s = get_settings()
    migrate(s.database_url_lg)
    conn = psycopg.connect(s.database_url_lg, autocommit=True)
    conn.execute(
        "TRUNCATE checkpoints, checkpoint_writes, checkpoint_blobs, "
        "store, store_vectors CASCADE"
    )
    embedder = FakeEmbedder()
    # LangGraph gets its own pool; the app SQL (agent.conn) keeps the plain
    # connection, avoiding psycopg pipeline-state corruption.
    from psycopg_pool import ConnectionPool
    pool = ConnectionPool(conninfo=s.database_url_lg, min_size=1, max_size=4,
                          open=True)
    store = build_store(pool, embedder)
    check = build_checkpointer(pool)
    yield {"conn": conn, "store": store, "check": check, "embedder": embedder}
    conn.execute(
        "TRUNCATE checkpoints, checkpoint_writes, checkpoint_blobs, "
        "store, store_vectors CASCADE"
    )
    pool.close()
    conn.close()


def _agent(lg, user_id):
    return LangGraphAgent(
        user_id=user_id, chat=FakeChat(), embedder=lg["embedder"],
        conn=lg["conn"], store=lg["store"], checkpointer=lg["check"],
    )


def _uid():
    import random
    return f"lg-{random.randint(10**8, 10**9 - 1)}"


def test_fact_supersession_and_cross_thread(lg):
    """put()-overwrite supersession: the latest value wins, across threads."""
    u = _uid()
    ag = _agent(lg, u)
    ag._write_facts([{"kind": "applicant", "name": "SELF",
                      "attr": "income_monthly", "value": 7900}])
    ag._write_facts([{"kind": "applicant", "name": "SELF",
                      "attr": "income_monthly", "value": 8750}])
    assert ag._resolve("applicant", u) == {"income_monthly": 8750}
    # a different thread reads the same user-scoped store
    ag2 = _agent(lg, u)
    assert ag2._resolve("applicant", u) == {"income_monthly": 8750}


def test_scope_isolation(lg):
    u1, u2 = _uid(), _uid()
    a1 = _agent(lg, u1)
    a1._write_facts([{"kind": "applicant", "name": "SELF",
                      "attr": "income_monthly", "value": 8000}])
    a2 = _agent(lg, u2)
    assert a2._resolve("applicant", u2) is None
    assert a1._resolve("applicant", u1) == {"income_monthly": 8000}


def test_turn_recall_and_recent_window(lg):
    u = _uid()
    ag = _agent(lg, u)
    ag._store_turn("th1", 1, "my income is 8750")
    ag._store_turn("th1", 3, "the max LTV is 75")
    hits = ag._recall("th1", "my income is 8750")
    assert hits, "stored turns should be searchable"
    assert hits[0]["content"] == "my income is 8750"

    msgs = [HumanMessage(content="hi"), AIMessage(content="hello"),
            HumanMessage(content="now ask")]
    recent = ag._recent(msgs)
    # excludes the current (last) message; keeps role mapping
    assert [t["content"] for t in recent] == ["hi", "hello"]
    assert [t["role"] for t in recent] == ["user", "assistant"]


def test_persona_ensure_idempotent(lg):
    u = _uid()
    ag = _agent(lg, u)
    v1 = ag._get_persona()
    assert v1 is not None and v1["version"] == 1
    ag._ensure_persona()
    assert ag._get_persona()["version"] == 1


def test_one_turn_graph_round_trip(lg):
    """Locks the config/thread_id wiring: one invoke produces the outcome."""
    u = _uid()
    ag = _agent(lg, u)
    ag._write_facts([{"kind": "applicant", "name": "SELF",
                      "attr": "income_monthly", "value": 8750}])
    out = ag.answer(thread_id="t1", text="What is my income?")
    assert out.text == "8,750"
    assert out.facts_used["you"]["applicant.income_monthly"] == 8750
    assert out.thread_id == "t1"
