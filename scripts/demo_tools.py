"""Unit 5, live: the planner loop, fresh vs cache, and the offload.

Part 1 + 2 go through MemoryAgent, so the PLANNER decides the tools.
Part 3 calls the toolbox directly to make reuse/offload visible regardless
of the planner's choices that day.
"""

import random
import sys

import psycopg

from agentmem.agent import MemoryAgent
from agentmem.db import migrate
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import toolbox


def main() -> int:
    s = get_settings()
    if not s.has_api_key:
        print("MEM_API_KEY not set")
        return 1
    migrate(s.database_url)
    conn = psycopg.connect(s.database_url, autocommit=True)
    user = f"tools-demo-{random.randint(10**8, 10**9 - 1)}"
    agent = MemoryAgent(conn, user_id=user, chat=ChatClient(), embedder=Embedder())

    q_pay = ("I make 6,800 a month and my other debt payments are 1,200 "
             "monthly. Under Acme's DTI limit, what's the most I can "
             "qualify to pay per month overall, and what does that leave?")
    q_rate = "What's the current rate on a 30-year fixed with a 60-day lock?"

    print("=" * 76)
    print("Q:", q_pay)
    out1 = agent.answer(thread_id="tool-thread", text=q_pay)
    print(f"[tools used: {out1.tool_calls}]")
    print(out1.text)

    print("=" * 76)
    print("Q:", q_rate)
    out2 = agent.answer(thread_id="tool-thread", text=q_rate)
    print(f"[tools used: {out2.tool_calls}]")
    print(out2.text)

    print("=" * 76)
    print("[DIRECT TOOLBOX DEMO] fresh vs cache vs offload")
    a = toolbox.run_tool(conn, user_id=user, thread_id="direct",
                         name="rate_lookup",
                         args={"term_years": 15, "lock_days": 30})
    b = toolbox.run_tool(conn, user_id=user, thread_id="direct",
                         name="rate_lookup",
                         args={"term_years": 15, "lock_days": 30})
    d = toolbox.run_tool(conn, user_id=user, thread_id="direct",
                         name="policy_full_dump", args={})
    print(f"rate 15y/30d first : source={a['source']} bytes={a['result_bytes']}")
    print(f"rate 15y/30d again : source={b['source']} (identical args => reuse)")
    print(f"policy_full_dump   : source={d['source']} bytes={d['result_bytes']} "
          f"offloaded={d['offloaded']}")
    print(f"log for {user}:")
    for t in toolbox.recent_calls(conn, user_id=user, limit=5):
        print(f"  {t['created_at']:%H:%M:%S} {t['tool']:<26} {t['source']:<6} {t['bytes']}B")

    ok = (out1.tool_calls and out1.tool_calls[0]["name"] == "affordability_calculator"
          and any(t["source"] == "cache" for t in [b]) and d["offloaded"])
    for table in ("mem.turn", "mem.fact", "mem.entity", "mem.persona",
                  "mem.tool_call"):
        conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (user,))
    conn.close()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
