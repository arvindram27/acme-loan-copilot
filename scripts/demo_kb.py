"""Milestones 2-4, live: cited policy answers + same-turn fact correction.

Demo A (citations): the baseline has no corpus and must hedge; the memory
agent retrieves the LTV policy and should answer 75% citing the doc.

Demo B (extraction + supersession in one turn): the agent is told a NEW
income in the same turn it is asked; validated extraction writes the fact
(superseding the old one) BEFORE context assembly, so the answer reflects
the correction immediately.
"""

import random
import sys

import psycopg

from agentmem.agent import BaselineAgent, MemoryAgent
from agentmem.db import migrate
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import entity

Q_LTV = "What is the maximum LTV on a two-unit investment property purchase?"


def main() -> int:
    if get_settings().has_api_key is False:
        print("MEM_API_KEY not set")
        return 1
    s = get_settings()
    migrate(s.database_url)
    conn = psycopg.connect(s.database_url, autocommit=True)
    user = f"kb-demo-{random.randint(10**8, 10**9 - 1)}"
    chat = ChatClient()
    baseline = BaselineAgent(chat).say(Q_LTV)

    agent = MemoryAgent(conn, user_id=user, chat=chat, embedder=Embedder())
    out_ltv = agent.answer(thread_id="kb-thread", text=Q_LTV)

    seeded_income = 70000
    entity.write_fact(conn, user_id=user, thread_id="kb-thread",
                      kind="applicant", name="Ravi Chandra",
                      attr="income_annual", value=seeded_income)
    q = ("Actually, Ravi Chandra's annual income is now 84,000. "
         "Given that, what is his annual income?")
    out_corr = agent.answer(thread_id="kb-thread", text=q,
                            entity_refs=[("applicant", "Ravi Chandra")])

    print("=" * 76)
    print("[BASELINE, no corpus]")
    print(baseline)
    print("=" * 76)
    print(f"[MEMORY AGENT + KB] retrieved_chunks={len(out_ltv.retrieved_refs)} "
          f"tokens={out_ltv.tokens}")
    print(out_ltv.text)
    print("=" * 76)
    print(f"[EXTRACTION DEMO] facts_written={out_corr.facts_written} "
          f"dropped={out_corr.dropped_facts}")
    print(out_corr.text)
    print("=" * 76)

    trail = entity.fact_history(
        conn,
        entity.resolve(conn, user_id=user, kind="applicant",
                       name="Ravi Chandra")["entity_id"],
        "income_annual",
    )
    print("income fact trail:",
          [(t["value"], t["status"]) for t in trail])

    ok = ("75" in out_ltv.text) and any(
        t["value"] == 84000 and t["status"] == "active" for t in trail
    )
    for table in ("mem.turn", "mem.fact", "mem.entity", "mem.persona"):
        conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user,))
    conn.close()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
