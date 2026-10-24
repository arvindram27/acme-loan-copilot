"""FROZEN held-out set v1 — questions, expectations and scoring, fixed before
the first run.

Protocol (review 4, evaluation item):

  1. This file is committed with **no results** (commit F1). That freezes the
     questions, the expected answers and the scoring rules under a commit hash.
  2. Only then is it run, exactly once.
  3. The first result — including failures — is committed as-is (commit F2).
     The artifact records F1's hash, so a reader can verify the order.
  4. v1 is never edited after the run. If a failure reveals a product bug, the
     fix lands separately and any re-evaluation is a NEW frozen set (v2).

The set is deliberately *new*: none of these questions or numbers appear in
WORKLOAD_V1 or in scripts/heldout_check.py.

Scoring (frozen):

    pass(q) = all(must_include present)
              and all(must_exclude absent)
              and (cite ? "§" in answer and retrieved_refs non-empty : true)
              and (require_tool ? that tool appears in tool_calls : true)

  * "present"/"absent" use agentmem.eval.numcompare.contains_value — the same
    numeric comparator the graders use: fuzzy (half of the last decimal place)
    for numbers, so a headline "2,430" matches 2430. No substring matching.
  * must_exclude is strict: the value must not appear anywhere in the answer.
    For a "what is on file" question the correct answer is just the current
    value, so volunteering the superseded number is a failure even if it is
    labelled as history.
  * cite means the answer must carry a policy citation marker ("§") and the turn
    must actually have retrieved KB chunks.
  * require_tool means the answer must have used that tool, not merely produced
    the right number.

Run (requires the KB seeded, e.g. `python scripts/seed_kb.py`):

    python scripts/heldout_frozen.py
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from agentmem.agent import MemoryAgent
from agentmem.db import migrate
from agentmem.eval.numcompare import contains_value
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import entity

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs" / "evidence" / "heldout-frozen.json"
OUT_TXT = ROOT / "docs" / "evidence" / "heldout-frozen.txt"

# Seeded once, before question 1. New numbers: income 7,400, debts 900.
SEED_MESSAGE = (
    "I make 7,400 a month, and 900 of that goes to my car payment."
)

QUESTIONS = [
    {
        "id": "h1-tool",
        "kind": "tool",
        "thread": "h",
        "question": (
            "With my income and my car payment, what is the most I could pay "
            "each month under Acme's DTI limit?"
        ),
        # 0.45 * 7,400 = 3,330; minus 900 in debts = 2,430.
        "must_include": ["2430"],
        "must_exclude": [],
        "cite": False,
        "require_tool": "affordability_calculator",
    },
    {
        "id": "h2-rag-ltv",
        "kind": "rag",
        "thread": "h",
        "question": (
            "What is Acme's maximum LTV for a two-unit owner-occupied property?"
        ),
        "must_include": ["85"],
        "must_exclude": [],
        "cite": True,
        "require_tool": None,
    },
    {
        "id": "h3-rag-lock",
        "kind": "rag",
        "thread": "h",
        "question": (
            "If my rate lock expires and I extend it beyond the grace period, "
            "what does that cost?"
        ),
        # 0.125 points per seven days. Only the rate is graded: the comparator
        # reads digits, and the answer may legitimately spell "seven days".
        "must_include": ["0.125"],
        "must_exclude": [],
        "cite": True,
        "require_tool": None,
    },
    {
        "id": "h4-supersede",
        "kind": "supersede",
        "thread": "h",
        "question": (
            "My raise came through - I make 8,100 a month now. What is my "
            "income on file?"
        ),
        "must_include": ["8100"],
        "must_exclude": ["7400"],
        "cite": False,
        "require_tool": None,
    },
    {
        "id": "h5-cross-thread",
        "kind": "cross_thread",
        "thread": "h2",
        "question": "What income and monthly debts are on file for me?",
        "must_include": ["8100", "900"],
        "must_exclude": ["7400"],
        "cite": False,
        "require_tool": None,
    },
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(q: dict, out) -> tuple[bool, dict]:
    """Apply the frozen rule above. Returns (passed, per-check detail)."""
    include = {v: contains_value(out.text, v) for v in q["must_include"]}
    exclude = {v: not contains_value(out.text, v) for v in q["must_exclude"]}
    cite_ok = (
        ("\u00a7" in out.text and bool(out.retrieved_refs))
        if q["cite"] else True
    )
    names = [c.get("name") for c in out.tool_calls]
    tool_ok = (q["require_tool"] in names) if q["require_tool"] else True
    passed = all(include.values()) and all(exclude.values()) and cite_ok and tool_ok
    return passed, {
        "must_include": include,
        "must_exclude_absent": exclude,
        "cite_ok": cite_ok,
        "tool_ok": tool_ok,
        "tool_calls": names,
    }


def main() -> int:
    s = get_settings()
    migrate(s.database_url)
    conn = psycopg.connect(s.database_url, autocommit=True)

    chunks = conn.execute("SELECT count(*) FROM mem.chunk").fetchone()[0]
    if not chunks:
        print("KB is empty: run `python scripts/seed_kb.py` first.", file=sys.stderr)
        return 2

    # Captured BEFORE the artifacts exist, so "clean" means the frozen commit
    # was the tip with no uncommitted changes.
    frozen_commit = _git("rev-parse", "--short", "HEAD")
    worktree_clean = _git("status", "--porcelain") == ""

    u = f"heldout-frozen-{random.randint(10**8, 10**9 - 1)}"
    rows: list[dict] = []
    lines: list[str] = []

    try:
        chat = ChatClient()
        embedder = Embedder()
        agent = MemoryAgent(conn, user_id=u, chat=chat, embedder=embedder)
        agent.answer(thread_id="h", text=SEED_MESSAGE)

        header = f"{'id':<16} {'kind':<13} {'result':<7} answer"
        print(header)
        lines.append(header)
        for q in QUESTIONS:
            out = agent.answer(thread_id=q["thread"], text=q["question"])
            passed, checks = score(q, out)
            flat = out.text.replace("\n", " ").strip()
            row = {
                "id": q["id"],
                "kind": q["kind"],
                "thread": q["thread"],
                "question": q["question"],
                "must_include": q["must_include"],
                "must_exclude": q["must_exclude"],
                "cite": q["cite"],
                "require_tool": q["require_tool"],
                "pass": passed,
                "checks": checks,
                "answer": out.text,
                "retrieved_refs": out.retrieved_refs,
                "tool_calls": out.tool_calls,
            }
            rows.append(row)
            line = f"{q['id']:<16} {q['kind']:<13} {'PASS' if passed else 'FAIL':<7} {flat[:88]}"
            print(line)
            lines.append(line)
    finally:
        entity.purge_user(conn, user_id=u)
        for table in ("mem.turn", "mem.persona", "mem.summary", "mem.cache",
                      "mem.tool_call", "mem.app_trace"):
            conn.execute(f"DELETE FROM {table} WHERE user_id = %s", (u,))
        conn.close()

    passed_n = sum(1 for r in rows if r["pass"])
    total = len(rows)
    summary_line = f"\n{passed_n}/{total} passed"
    print(summary_line)
    lines.append(summary_line)

    payload = {
        "_meta": {
            "set": "heldout-frozen-v1",
            "frozen_commit": frozen_commit,
            "frozen_file": "scripts/heldout_frozen.py",
            "frozen_file_sha256": _sha256(Path(__file__)),
            "worktree_clean_at_run": worktree_clean,
            "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chat_model": s.chat_model,
            "embed_model": s.embed_model,
            "seed_message": SEED_MESSAGE,
            "kb_chunks": chunks,
            "repeat_count": 1,
            "scoring": (
                "pass = all must_include present AND all must_exclude absent "
                "AND (cite ? '§' in answer and retrieved_refs non-empty : true) "
                "AND (require_tool ? tool in tool_calls : true); numeric "
                "presence via numcompare.contains_value (fuzzy, half of the "
                "last decimal place)"
            ),
            "note": (
                "v1 is frozen and never edited after this run; failures are "
                "retained. A re-evaluation must be a new frozen set (v2)."
            ),
        },
        "summary": {
            "passed": passed_n,
            "total": total,
            "all_pass": passed_n == total,
        },
        "questions": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_TXT.relative_to(ROOT)}")
    return 0 if passed_n == total else 1


if __name__ == "__main__":
    sys.exit(main())
