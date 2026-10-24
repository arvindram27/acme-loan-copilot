"""Per-turn chat-call/time budget (review #10)."""

import random

import psycopg
import pytest

from agentmem.agent import (
    MemoryAgent,
    TurnBudget,
    TurnBudgetExceeded,
    _BudgetedChat,
)
from agentmem.db import migrate
from agentmem.settings import get_settings
from agentmem.units import entity

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
    def __init__(self, final_text):
        self.final_text = final_text
        self.final_calls = 0

    def complete(self, messages, **kw):
        system = messages[0]["content"] if messages else ""

        class R:
            total_tokens = 3
            prompt_tokens = 2
            model = "fake"

        r = R()
        if "Extract factual statements" in system:
            r.text = '{"facts": []}'
        elif "tool-planning" in system or "decide the next step" in system:
            r.text = '{"action": "answer"}'
        else:
            self.final_calls += 1
            r.text = self.final_text
        return r


def test_turn_budget_exhausts_by_calls():
    b = TurnBudget(max_calls=2)
    assert b.available()
    b.record()
    b.record()
    assert not b.available()


def test_budgeted_chat_raises_when_spent():
    class _C:
        def complete(self, messages, **kw):
            return "ok"

    b = TurnBudget(max_calls=1)
    bc = _BudgetedChat(_C(), b)
    bc.complete([])
    with pytest.raises(TurnBudgetExceeded):
        bc.complete([])


@pytest.fixture()
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    entity.purge_user(c, pattern="tb-%")
    for table in ("mem.turn", "mem.persona", "mem.summary", "mem.cache"):
        c.execute(f"DELETE FROM {table} WHERE user_id LIKE 'tb-%'")
    c.close()


def test_budget_skips_optional_supersession_rewrite(conn):
    u = f"tb-{random.randint(10**8, 10**9 - 1)}"
    entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                      name=u, attr="income_monthly", value=7900)
    entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                      name=u, attr="income_monthly", value=8750)
    chat = FakeChat(final_text="your income is $7,900")  # stale draft
    # mandatory path: extraction(1) + planner(2) + final(3) = 3 calls; the
    # supersession rewrite would be call 4, so a budget of 3 forces a skip.
    agent = MemoryAgent(conn, user_id=u, chat=chat, embedder=FakeEmbedder(),
                        max_calls_per_turn=3)
    out = agent.answer(thread_id="t", text="What is my income on file?")
    assert out.text == "your income is $7,900", "stale draft kept, no rewrite"
    assert chat.final_calls == 1, "the optional rewrite never reached the client"
