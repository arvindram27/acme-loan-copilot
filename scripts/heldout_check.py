"""Held-out generalization check: 5 questions NOT in WORKLOAD_V1.

The supersession guard was added after seeing a failure on the 32-item workload,
so a fair question is whether it generalises. These five fresh, dataset-relevant
questions (different numbers, different phrasings) exercise the same memory
behaviours — tool arithmetic, policy grounding, supersession, cross-thread
recall — on data the workload never saw.

    python scripts/heldout_check.py
"""

from __future__ import annotations

import random
import sys

import psycopg

from agentmem.agent import MemoryAgent
from agentmem.db import migrate
from agentmem.eval.numcompare import contains_value
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import entity

SEED = ("I make 9,200 a month, and 1,100 of that goes to my student loans.",
        None)

QUESTIONS = [
    # (kind, thread, question, must_include, must_exclude)
    ("tool", "h",
     "With my income and my student-loan payment, what is the most I could pay "
     "each month under Acme's DTI limit?", ["3040"], []),
    ("rag_ltv", "h",
     "What is Acme's maximum LTV for a one-unit investment property?",
     ["80"], []),
    ("rag_fico", "h",
     "What is the minimum FICO Acme requires for a conventional loan?",
     ["620"], []),
    ("supersede", "h",
     "My raise came through - I make 9,600 a month now. What is my income on "
     "file?", ["9600"], ["9200"]),
    ("cross_thread", "h2",
     "What income and monthly debts are on file for me?", ["9600", "1100"], []),
]


def main() -> int:
    s = get_settings()
    migrate(s.database_url)
    conn = psycopg.connect(s.database_url, autocommit=True)
    u = f"heldout-{random.randint(10**8, 10**9 - 1)}"

    try:
        chat = ChatClient()
        embedder = Embedder()
        agent = MemoryAgent(conn, user_id=u, chat=chat, embedder=embedder)
        agent.answer(thread_id="h", text=SEED[0])

        all_ok = True
        print(f"{'kind':<12} {'pass':<5} {'notes':<8} answer")
        for kind, thread, q, inc, exc in QUESTIONS:
            out = agent.answer(thread_id=thread, text=q)
            ok_inc = all(contains_value(out.text, v) for v in inc)
            ok_exc = all(not contains_value(out.text, v) for v in exc)
            cited = bool(out.retrieved_refs) and ("\u00a7" in out.text)
            if kind.startswith("rag"):
                ok_inc = ok_inc and cited
            ok = ok_inc and ok_exc
            all_ok = all_ok and ok
            notes = "cite" if (kind.startswith("rag") and cited) else ""
            print(f"{kind:<12} {'PASS' if ok else 'FAIL':<5} {notes:<8} "
                  f"{out.text.replace(chr(10), ' ')[:80]}")
    finally:
        entity.purge_user(conn, user_id=u)
        for table in ("mem.turn", "mem.persona", "mem.summary", "mem.cache"):
            conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (u,))
        conn.close()

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
