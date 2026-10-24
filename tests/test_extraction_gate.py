"""Extraction gate: with extract_turns=False, no extraction LLM call is made."""

import random

import psycopg
import pytest

from agentmem.agent import MemoryAgent
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


class CountingChat:
    def __init__(self):
        self.extraction_calls = 0

    def complete(self, messages, **kw):
        system = messages[0]["content"] if messages else ""

        class R:
            total_tokens = 3
            prompt_tokens = 2
            model = "fake"

        r = R()
        if "Extract factual statements" in system:
            self.extraction_calls += 1
            r.text = ('{"facts": [{"kind": "applicant", "name": "SELF", '
                      '"attr": "income_monthly", "value": 9000}]}')
        elif "tool-planning" in system or "decide the next step" in system:
            r.text = '{"action": "answer"}'
        else:
            r.text = "ok"
        return r


@pytest.fixture()
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    c.close()


def _cleanup(conn, user):
    for table in ("mem.turn", "mem.fact", "mem.entity", "mem.persona",
                  "mem.summary", "mem.cache"):
        conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user,))


def test_extraction_skipped_when_disabled(conn):
    u = f"gate-{random.randint(10**8, 10**9 - 1)}"
    chat = CountingChat()
    agent = MemoryAgent(conn, user_id=u, chat=chat, embedder=FakeEmbedder(),
                        extract_turns=False)
    try:
        agent.answer(thread_id="t", text="I make 9,000 a month now.")
        assert chat.extraction_calls == 0, "no extraction call when disabled"
        n = conn.execute("SELECT count(*) FROM mem.fact WHERE user_id = %s",
                         (u,)).fetchone()[0]
        assert n == 0, "no fact written when extraction is disabled"
    finally:
        _cleanup(conn, u)


def test_extraction_runs_when_enabled(conn):
    """Control: with the flag on the call happens and the fact lands."""
    u = f"gate-{random.randint(10**8, 10**9 - 1)}"
    chat = CountingChat()
    agent = MemoryAgent(conn, user_id=u, chat=chat, embedder=FakeEmbedder(),
                        extract_turns=True)
    try:
        agent.answer(thread_id="t", text="I make 9,000 a month now.")
        assert chat.extraction_calls == 1
        from agentmem.units import entity
        facts = entity.resolve(conn, user_id=u, kind="applicant", name=u)
        assert facts is not None and facts["facts"]["income_monthly"] == 9000
    finally:
        _cleanup(conn, u)
