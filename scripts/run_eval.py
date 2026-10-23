"""Run the golden workload end-to-end and persist + report.

Backends:
  memory    MemoryAgent (recency + recall + entity + KB + toolbox)
  baseline  BaselineAgent (no memory, no tools, no KB) — the contrast

For each turn it records latency, tokens, cost, tool-cache hits and the full
TurnOutcome, then grades the answer with the cascade (code always; Jev and
the judge only when MEM_JEV_KEY is set). Everything lands in mem.eval_run /
eval_item / grade so the milestone-8 app can replay it.

Usage:
    python scripts/run_eval.py [--backend memory,baseline] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import psycopg

from agentmem.agent import BaselineAgent, MemoryAgent
from agentmem.db import migrate
from agentmem.eval.graders import JevGrader, LLMJudge, run_cascade
from agentmem.eval.metrics import summarize
from agentmem.eval.workload import WORKLOAD_V1
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.providers.jev import JevClient
from agentmem.settings import get_settings
from agentmem.units import knowledge, toolbox

# Approximate blended $/1k tokens (gpt-4o-mini in+out average) for the cost
# line; the eval reports it as an estimate, not an invoice.
USD_PER_1K_TOKENS = 0.0004


def _seed_kb(conn, embedder) -> None:
    from pathlib import Path
    n = conn.execute("SELECT COUNT(*) FROM mem.chunk").fetchone()[0]
    if n:
        return
    corpus = json.loads(
        (Path(__file__).resolve().parents[1]
         / "infra" / "seed" / "corpus.json").read_text(encoding="utf-8")
    )
    for doc in corpus["docs"]:
        knowledge.add_doc(conn, title=doc["title"],
                          version=corpus["version"], text=doc["text"],
                          embedding_fn=embedder.embed)


def _outcome_dict(outcome) -> dict:
    if outcome is None:
        return {}
    return {
        "text": outcome.text,
        "thread_id": outcome.thread_id,
        "recent_turns": outcome.recent_turns,
        "recalled_turns": outcome.recalled_turns,
        "facts_used": outcome.facts_used,
        "facts_written": outcome.facts_written,
        "dropped_facts": outcome.dropped_facts,
        "retrieved_refs": outcome.retrieved_refs,
        "tool_calls": outcome.tool_calls,
        "tool_results": outcome.tool_results,
        "tokens": outcome.tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="memory,baseline")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]

    s = get_settings()
    migrate(s.database_url)
    conn = psycopg.connect(s.database_url, autocommit=True)
    embedder = Embedder()
    _seed_kb(conn, embedder)

    # Deterministic from scratch: clear the two workload users' memory so a
    # re-run measures the same thing (eval_run/eval_item history is kept).
    for table in ("mem.tool_call", "mem.turn", "mem.fact", "mem.entity",
                  "mem.persona"):
        conn.execute(f"DELETE FROM {table} WHERE user_id IN ('maya', 'liam')")

    jev_g = judge_g = None
    if s.jev_key:
        jev_g = JevGrader(JevClient(s), s.jev_escalate_low, s.jev_escalate_high)
        judge_g = LLMJudge(ChatClient())
        print(f"Jev grader ON ({s.jev_model})")

    steps = WORKLOAD_V1 if not args.limit else WORKLOAD_V1[:args.limit]

    for backend in backends:
        chat = ChatClient()
        agents: dict[str, MemoryAgent | BaselineAgent] = {}
        run_id = conn.execute(
            "INSERT INTO mem.eval_run (backend, model, workload) "
            "VALUES (%s, %s, %s) RETURNING run_id",
            (backend, s.chat_model, "workload_v1"),
        ).fetchone()[0]

        results = []
        for step in steps:
            user, thread, text = step["user"], step["thread"], step["text"]
            if backend == "memory":
                if user not in agents:
                    agents[user] = MemoryAgent(
                        conn, user_id=user, chat=chat, embedder=embedder)
                agent = agents[user]
                t0 = time.perf_counter()
                outcome = agent.answer(thread_id=thread, text=text)
                latency_ms = (time.perf_counter() - t0) * 1000
                tokens = outcome.tokens
            else:
                agent = BaselineAgent(chat)
                before = sum(chat.total_usage.values())
                t0 = time.perf_counter()
                out = agent.say(text)
                latency_ms = (time.perf_counter() - t0) * 1000
                outcome = None
                tokens = sum(chat.total_usage.values()) - before

            expected = step.get("expected") or {}
            item_id = conn.execute(
                "INSERT INTO mem.eval_item "
                "(run_id, item_no, user_id, thread_id, question, expected, "
                " outcome) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING item_id",
                (run_id, step["n"], user, thread, text,
                 json.dumps(expected, default=str),
                 json.dumps(_outcome_dict(outcome), default=str)),
            ).fetchone()[0]

            answer_text = outcome.text if outcome else out
            # grade with a duck-typed outcome; None text => fail everything
            if outcome is None:
                from agentmem.eval.graders import Outcome
                grade_oc = Outcome(text=answer_text)
            else:
                grade_oc = outcome
            grades = run_cascade(
                step["n"], expected, grade_oc, text,
                jev=jev_g, judge=judge_g,
            )

            code_fail = [g for g in grades
                         if g.grader == "code" and g.pass_ is False]
            answer_ok = not code_fail
            op_cats = {"entity_update", "entity_update_setup",
                       "same_turn_correction", "cross_thread_recall",
                       "recall", "scope"}
            for g in grades:
                conn.execute(
                    "INSERT INTO mem.grade (item_id, grader, check_name, "
                    " pass, score, probability, confidence, escalated, "
                    " rationale, latency_ms, cost_usd) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (item_id, g.grader, g.check, g.pass_,
                     g.value if isinstance(g.value, (int, float)) else None,
                     g.probability, g.confidence, g.escalated,
                     g.rationale, g.latency_ms, g.cost_usd),
                )
            results.append({
                "answer_ok": answer_ok,
                "op_ok": answer_ok if step.get("category") in op_cats else None,
                "latency_ms": latency_ms,
                "tokens": tokens,
                "cost_usd": tokens * USD_PER_1K_TOKENS / 1000,
                "cache_hit": any(
                    t.get("source") == "cache" for t in
                    (outcome.tool_calls if outcome else [])),
                "cache_eligible": any(
                    t.get("name") == "rate_lookup" for t in
                    (outcome.tool_calls if outcome else [])),
            })
            print(f"[{backend:<8}] n={step['n']:<2} "
                  f"{'OK ' if answer_ok else 'FAIL'} "
                  f"{latency_ms:6.0f}ms tok={tokens} "
                  f"tools={[t.get('name') for t in (outcome.tool_calls if outcome else [])] or '—'}")

        summ = summarize(results)
        conn.execute(
            "UPDATE mem.eval_run SET n_items=%s, metrics=%s, "
            " finished_at=now(), jev_cost_usd=%s WHERE run_id=%s",
            (len(results), json.dumps(summ.as_dict(), default=str),
             (jev_g.jev.total_cost_usd if jev_g else 0), run_id),
        )
        d = summ.as_dict()
        print(f"\n[{backend}] accuracy={d['accuracy']:.2f} "
              f"(95% CI {d['accuracy_ci'][0]:.2f}-{d['accuracy_ci'][1]:.2f}) "
              f"op_score={d['op_score']:.2f} cache_hit={d['cache_hit_rate']:.2f} "
              f"p50={d['p50_ms']:.0f}ms p95={d['p95_ms']:.0f}ms "
              f"tokens_mean={d['tokens_mean']:.0f} cost~${d['cost_usd']:.3f}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
