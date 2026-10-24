"""Live MemoryAgent test: persona + entity facts flow into the answer.

Skipped without MEM_API_KEY. Uses the real database and one real chat call.
"""

from agentmem.agent import MemoryAgent
from agentmem.db import migrate
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import entity

import psycopg
import pytest
import random

s = get_settings()
requires = pytest.mark.skipif(not s.has_api_key, reason="MEM_API_KEY not set")


@requires
def test_memory_agent_answers_from_facts():
    migrate(s.database_url)
    conn = psycopg.connect(s.database_url, autocommit=True)
    user = f"t-{random.randint(10**8, 10**9 - 1)}"
    try:
        entity.write_fact(conn, user_id=user, thread_id="t1", kind="applicant",
                          name="Jay Nathan", attr="income_monthly", value=8750)
        agent = MemoryAgent(conn, user_id=user, chat=ChatClient(),
                            embedder=Embedder())
        out = agent.answer(
            thread_id="live-thread",
            text="What is Jay Nathan's monthly income? Answer with the number.",
            entity_refs=[("applicant", "Jay Nathan")],
        )
        assert "8750" in out.text.replace(",", "").replace("$", "")
        assert out.facts_used, "facts should be attached to the outcome"
        seqs = {t["seq"] for t in __import__(
            "agentmem.units.conversation", fromlist=["x"]
        ).recent_turns(conn, "live-thread", k=10)}
        assert len(seqs) >= 2, "user and assistant turns persisted"
    finally:
        for table in ("mem.turn", "mem.persona"):
            conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user,))
        entity.purge_user(conn, user_id=user)
        conn.close()


pytestmark = [pytest.mark.db, pytest.mark.live]
