"""Regression run: the five *known* frozen questions, once, under scorer v2.

This is **not** a new held-out set and **not** an unseen holdout. The questions
are the ones frozen at `b92f0b1` and already published with their v1 result; they
are re-run here only to show behaviour under the stricter scorer and the current
code. The result is labelled `regression` and saved beside — never over — the
original.

The question set is loaded from the frozen script rather than copied, and the
model configuration is checked against the one recorded for v1; if it differs
the script stops instead of silently comparing two variables.

    python scripts/heldout_regression.py
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
from agentmem.eval.heldout_scorer import (
    SCORER_V2,
    Answer,
    load_frozen,
    score,
)
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings
from agentmem.units import entity

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ARTIFACT = ROOT / "docs" / "evidence" / "heldout-frozen.json"
OUT_JSON = ROOT / "docs" / "evidence" / "heldout-regression-v2.json"
OUT_TXT = ROOT / "docs" / "evidence" / "heldout-regression-v2.txt"

#: gpt-4o-mini list prices per 1M tokens, used only for the cost estimate.
CHAT_INPUT_USD_PER_M = 0.15
CHAT_OUTPUT_USD_PER_M = 0.60


class CountingChat:
    """Delegates to the real client and accumulates token usage."""

    def __init__(self, chat):
        self._chat = chat
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def usage(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def complete(self, messages, **kw):
        out = self._chat.complete(messages, **kw)
        self.calls += 1
        prompt = getattr(out, "prompt_tokens", 0) or 0
        total = getattr(out, "total_tokens", 0) or 0
        self.prompt_tokens += prompt
        self.total_tokens += total
        self.completion_tokens += max(total - prompt, 0)
        return out


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _delta(after: dict, before: dict) -> dict:
    return {k: after[k] - before[k] for k in after}


def main() -> int:
    s = get_settings()
    frozen = json.loads(FROZEN_ARTIFACT.read_text(encoding="utf-8"))
    v1_meta = frozen["_meta"]

    if s.chat_model != v1_meta["chat_model"] or s.embed_model != v1_meta["embed_model"]:
        print(
            "model configuration differs from the v1 record:\n"
            f"  v1: {v1_meta['chat_model']} / {v1_meta['embed_model']}\n"
            f"  now: {s.chat_model} / {s.embed_model}\n"
            "A different configuration needs a decision first, not a silent run.",
            file=sys.stderr,
        )
        return 3

    frozen_mod = load_frozen()
    migrate(s.database_url)
    conn = psycopg.connect(s.database_url, autocommit=True)

    chunks = conn.execute("SELECT count(*) FROM mem.chunk").fetchone()[0]
    if not chunks:
        print("KB is empty: run `python scripts/seed_kb.py` first.", file=sys.stderr)
        return 2

    command = "python scripts/heldout_regression.py"
    commit = _git("rev-parse", "--short", "HEAD")
    worktree_clean = _git("status", "--porcelain") == ""
    questions_sha = hashlib.sha256(
        Path(frozen_mod.__file__).read_bytes()
    ).hexdigest()

    u = f"regression-v2-{random.randint(10**8, 10**9 - 1)}"
    rows: list[dict] = []
    lines: list[str] = []

    try:
        chat = CountingChat(ChatClient())
        embedder = Embedder()
        agent = MemoryAgent(conn, user_id=u, chat=chat, embedder=embedder)

        before = chat.usage()
        emb_before = embedder.calls
        agent.answer(thread_id="h", text=frozen_mod.SEED_MESSAGE)
        seed_usage = _delta(chat.usage(), before)
        seed_embeds = embedder.calls - emb_before

        header = f"{'id':<16} {'result':<7} {'calls':<6} {'tokens':<7} answer"
        print(header)
        lines.append(header)

        for q in frozen_mod.QUESTIONS:
            b, e = chat.usage(), embedder.calls
            out = agent.answer(thread_id=q["thread"], text=q["question"])
            usage = _delta(chat.usage(), b)
            verdict = score(q, Answer.from_outcome(out), version=SCORER_V2)
            flat = out.text.replace("\n", " ").strip()

            rows.append({
                "id": q["id"],
                "kind": q["kind"],
                "thread": q["thread"],
                "question": q["question"],
                "must_include": q["must_include"],
                "must_exclude": q["must_exclude"],
                "cite": q["cite"],
                "require_tool": q["require_tool"],
                "pass": verdict.passed,
                "verdict": verdict.to_dict(),
                "answer": out.text,
                "retrieved_refs": out.retrieved_refs,
                "tool_calls": out.tool_calls,
                "chat_calls": usage["calls"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "embedding_calls": embedder.calls - e,
                "latency_ms": None,
            })
            line = (f"{q['id']:<16} {'PASS' if verdict.passed else 'FAIL':<7} "
                    f"{usage['calls']:<6} {usage['total_tokens']:<7} {flat[:70]}")
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
    usage = chat.usage()
    cost = (
        usage["prompt_tokens"] / 1_000_000 * CHAT_INPUT_USD_PER_M
        + usage["completion_tokens"] / 1_000_000 * CHAT_OUTPUT_USD_PER_M
    )
    failed = [r["id"] for r in rows if not r["pass"]]
    summary = (f"\n{passed_n}/{total} passed under {SCORER_V2}"
               + (f"  (failed: {', '.join(failed)})" if failed else ""))
    print(summary)
    lines.append(summary)
    print(f"chat calls={usage['calls']} tokens={usage['total_tokens']} "
          f"cost~${cost:.6f}")

    payload = {
        "_meta": {
            "kind": "regression",
            "label": (
                "Known frozen questions re-run once under scorer v2. This is a "
                "regression check, NOT a new unseen holdout, and it does not "
                "replace the v1 result."
            ),
            "question_set": v1_meta.get("set"),
            "question_set_frozen_commit": v1_meta.get("frozen_commit"),
            "questions_source": "scripts/heldout_frozen.py",
            "questions_source_sha256": questions_sha,
            "scorer_version": SCORER_V2,
            "command": command,
            "commit": commit,
            "worktree_clean_at_run": worktree_clean,
            "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chat_model": s.chat_model,
            "embed_model": s.embed_model,
            "same_model_config_as_v1": True,
            "seed_message": frozen_mod.SEED_MESSAGE,
            "kb_chunks": chunks,
            "repeat_count": 1,
            "token_usage": {
                "chat": usage,
                "seed_turn": seed_usage,
                "seed_embedding_calls": seed_embeds,
                "embedding_calls_total": embedder.calls,
                "note": (
                    "chat tokens cover every completion in the run (extraction, "
                    "planner, draft, retries) via a counting wrapper. Embedding "
                    "token counts are not returned by the client, so only call "
                    "counts are recorded."
                ),
            },
            "cost_estimate_usd": round(cost, 6),
            "cost_estimate_note": (
                f"chat only, at ${CHAT_INPUT_USD_PER_M}/1M input and "
                f"${CHAT_OUTPUT_USD_PER_M}/1M output (gpt-4o-mini list price at "
                "the time of writing); embeddings excluded (a few tenths of a "
                "cent); estimate, not billing."
            ),
        },
        "summary": {
            "passed": passed_n,
            "total": total,
            "all_pass": passed_n == total,
            "failed": failed,
        },
        "questions": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_TXT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
