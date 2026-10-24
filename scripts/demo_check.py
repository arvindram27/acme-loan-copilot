"""Demo acceptance check: five very different questions against both backends.

Resets the demo users, re-seeds a clean scenario (income 8,000 -> 8,750, car
loan 1,200), then asks five questions of five different kinds (policy/RAG,
arithmetic/tool, entity supersession, cross-thread recall, scope isolation) on
each backend and asserts the answer. Exits non-zero if any cell fails.

    python scripts/demo_check.py
"""

from __future__ import annotations

import json
import sys

import psycopg

from agentmem.app.deps import get_langgraph, make_agent
from agentmem.db import migrate
from agentmem.settings import get_settings

SEED = [
    ("maya", "demo", "I make 8,000 a month, and 1,200 of that goes to my car loan payment."),
    ("maya", "demo", "Quick update - my raise came through. I make 8,750 a month now."),
]

QUESTIONS = [
    # (kind, user, thread, question, must_include, must_exclude)
    ("policy", "maya", "demo",
     "If I buy a 2-unit investment property, what's the most I can borrow relative to its value?",
     ["75"], []),
    ("tool", "maya", "demo",
     "With my income and my car payment, what's the most I could pay each month under Acme's DTI limit?",
     ["2737"], []),
    ("entity", "maya", "demo",
     "What monthly income do you have on file for me?",
     ["8750"], ["8000"]),
    ("cross_thread", "maya", "demo-b",
     "Remind me - what income and monthly debts are on file for me?",
     ["8750", "1200"], []),
    ("scope", "liam", "x",
     "Is there another applicant named Maya? What's her monthly income?",
     [], ["8750"]),
]


def _norm(t: str) -> str:
    return (t.replace(",", "").replace("$", "").replace("%", "")
             .replace(" ", "").replace(".", "").lower())


def _trace(out) -> dict:
    return {
        "prompt_tokens": out.prompt_tokens,
        "context_tokens": out.context_tokens,
        "tokens": out.tokens,
        "recent_turns": out.recent_turns,
        "recalled_turns": out.recalled_turns,
        "summary_used": out.summary_used,
        "summary_turns": out.summary_turns,
        "answer_cache_hit": out.answer_cache_hit,
        "facts_used": out.facts_used,
        "retrieved_refs": [r.get("title") for r in out.retrieved_refs],
        "tools": [t.get("name") for t in out.tool_calls],
        "tool_sources": [t.get("source") for t in out.tool_calls],
    }


def _save_trace(conn, user, thread, backend, seq, question, out) -> None:
    conn.execute(
        "INSERT INTO mem.app_trace "
        "(user_id, thread_id, backend, seq, question, answer, trace) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user, thread, backend, seq, question, out.text,
         json.dumps(_trace(out), default=str)),
    )


def _next_seq(conn, user, thread, backend) -> int:
    return conn.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM mem.app_trace "
        "WHERE user_id = %s AND thread_id = %s AND backend = %s",
        (user, thread, backend),
    ).fetchone()[0]


def run_backend(backend: str, conn) -> list[dict]:
    results: list[dict] = []

    # seed (also recorded so the web chat shows the full conversation)
    agent = make_agent(backend, "maya", conn)
    for user, thread, text in SEED:
        out = agent.answer(thread_id=thread, text=text)
        _save_trace(conn, user, thread, backend,
                    _next_seq(conn, user, thread, backend), text, out)

    # the five questions
    for kind, user, thread, q, inc, exc in QUESTIONS:
        agent = make_agent(backend, user, conn)
        out = agent.answer(thread_id=thread, text=q)
        _save_trace(conn, user, thread, backend,
                    _next_seq(conn, user, thread, backend), q, out)

        n = _norm(out.text)
        ok_inc = all(_norm(v) in n for v in inc)
        ok_exc = all(_norm(v) not in n for v in exc)

        notes: list[str] = []
        if kind == "policy":
            cited = bool(out.retrieved_refs) and ("\u00a7" in out.text)
            notes.append("cite" if cited else "NO-CITE")
            ok_inc = ok_inc and cited
        if kind == "tool":
            called = any(t.get("name") == "affordability_calculator"
                         for t in out.tool_calls)
            notes.append("tool" if called else "NO-TOOL")
            ok_inc = ok_inc and called

        ok = ok_inc and ok_exc
        results.append({
            "kind": kind, "ok": ok, "notes": " ".join(notes),
            "answer": out.text.replace("\n", " ")[:90],
        })
    return results


def main() -> int:
    s = get_settings()
    migrate(s.database_url)
    migrate(s.database_url_lg)

    conn = psycopg.connect(s.database_url, autocommit=True)

    # build langgraph resources first (creates store/checkpointer tables + KB)
    lg = get_langgraph()

    # reset scratch demo users
    for table in ("turn", "fact", "entity", "persona", "summary", "cache",
                  "tool_call", "app_trace"):
        conn.execute(f"DELETE FROM mem.{table} WHERE user_id IN ('maya', 'liam')")

    # reset langgraph store/checkpointer + tool log
    lg["conn"].execute(
        "TRUNCATE checkpoints, checkpoint_writes, checkpoint_blobs, "
        "store, store_vectors CASCADE"
    )
    lg["conn"].execute(
        "DELETE FROM mem.tool_call WHERE user_id IN ('maya', 'liam')"
    )

    all_ok = True
    for backend in ("scratch", "langgraph"):
        results = run_backend(backend, conn)
        print(f"\n=== {backend} ===")
        print(f"{'kind':<13} {'pass':<5} {'notes':<10} answer")
        for r in results:
            all_ok = all_ok and r["ok"]
            print(f"{r['kind']:<13} {'PASS' if r['ok'] else 'FAIL':<5} "
                  f"{r['notes']:<10} {r['answer']}")

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
