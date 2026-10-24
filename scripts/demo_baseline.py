"""Unit 0 vs Units 1-3, live: the reason agent memory exists.

One run, two agents, two threads:
  - BaselineAgent answers with nothing but its system prompt.
  - MemoryAgent answers with persona, entity facts and conversation recall.

Before the run, MemoryAgent's store is seeded "as if" an earlier interview
happened: an income fact plus a 12-turn interview transcript are written by
the app (the production path that extracts facts from the interview is built
in the next milestone step).

Expected: the baseline cannot know Jane's income; the memory agent answers
105,000. Both calls cost a few cents of gpt-4o-mini.
"""

import random
import sys

import psycopg

from agentmem.agent import BaselineAgent, MemoryAgent
from agentmem.db import migrate
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import conversation, entity

QUESTION = ("Without me repeating anything from today: what is Jane's "
            "current annual income, and what did she say about it early on?")


def seed(conn) -> tuple[str, str]:
    """Write an interview history and an income fact that was UPDATED."""
    user = f"demo-{random.randint(10 ** 8, 10 ** 9 - 1)}"
    thread = "interview"
    em = Embedder()

    transcript = [
        ("user", "Hi, I'm helping Jane Doe with a mortgage pre-approval."),
        ("assistant", "Working with you on Jane Doe's file. What can you tell me?"),
        ("user", "She's buying a two-unit investment property."),
        ("assistant", "Noted: two-unit investment property for Jane Doe."),
        ("user", "Actually, one more thing about Jane: she was worried about DTI."),
        ("assistant", "Keep her DTI context in mind for this file."),
        ("user", "Jane also mentioned her bonus structure varies by quarter."),
        ("assistant", "Recorded: variable bonus structure for Jane's income."),
        ("user", "Jane completed her probation at her new job in month 6."),
        ("assistant", "Employment notes taken for Jane."),
        ("user", "That's all from Jane for now, thanks."),
        ("assistant", "Ready when the file moves forward."),
    ]
    for i, (role, content) in enumerate(transcript, start=1):
        conversation.append_turn(
            conn, user_id=user, thread_id=thread, role=role, content=content,
            embedding=em.embed(content) if role == "user" else None,
        )

    # Entity memory: the income was stated, then corrected.
    entity.write_fact(conn, user_id=user, thread_id=thread, kind="applicant",
                      name="Jane Doe", attr="income_annual", value=95000,
                      source_ref="turn:3")
    entity.write_fact(conn, user_id=user, thread_id=thread, kind="applicant",
                      name="Jane Doe", attr="income_annual", value=105000,
                      source_ref="turn:12", source_kind="stated")
    entity.write_fact(conn, user_id=user, thread_id=thread, kind="property",
                      name="48 Larch St", attr="units", value=2)
    return user, thread


def main() -> int:
    migrate(get_settings().database_url)
    s = get_settings()
    if not s.has_api_key:
        print("MEM_API_KEY not set")
        return 1
    conn = psycopg.connect(s.database_url, autocommit=True)
    chat = ChatClient()
    user, thread = seed(conn)

    print(f"seeded user: {user}")
    print(f"question: {QUESTION}\n")

    baseline = BaselineAgent(chat).say(QUESTION)
    agent = MemoryAgent(conn, user_id=user, chat=chat, embedder=Embedder())
    n1 = agent.answer(thread_id="new-thread-fresh", text=QUESTION,
                      entity_refs=[("applicant", "Jane Doe")])
    n2 = agent.answer(thread_id=thread, text=QUESTION,
                      entity_refs=[("applicant", "Jane Doe")],
                      )

    print("=" * 72)
    print("[BASELINE (no memory)]")
    print(baseline)
    print("=" * 72)
    print("[MEMORY AGENT, brand-new thread] (persona + entity facts, no turns)")
    print(f"(recency={n1.recent_turns} recalled={n1.recalled_turns} "
          f"facts={n1.facts_used and 'yes'})")
    print(n1.text)
    print("=" * 72)
    print("[MEMORY AGENT, original interview thread] (with semantic recall)")
    print(f"(recency={n2.recent_turns} recalled={n2.recalled_turns})")
    print(n2.text)
    print("=" * 72)

    recall_used = "105" in n1.text or "105" in n2.text
    print("RESULT: memory agent produced the corrected income 105000:",
          recall_used)
    conn.execute("DELETE FROM mem.turn WHERE user_id = %s", (user,))
    entity.purge_user(conn, user_id=user)
    conn.execute("DELETE FROM mem.persona WHERE user_id = %s", (user,))
    conn.close()
    return 0 if recall_used else 1


if __name__ == "__main__":
    sys.exit(main())
