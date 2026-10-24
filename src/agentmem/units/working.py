"""Unit 6 — working memory: rolling summaries of a thread's older turns.

When a thread grows past the verbatim recency window, the turns that fall out
of it are folded into a compressed summary that records the turn seqs it
covers. A long interview therefore compacts instead of growing unbounded, and
the context builder (units/context.py) can cite the summary instead of raw
history.

When to use it: compressing older conversation so recency + summary + recall
fit a token budget.
What it is NOT for: durable facts (entity memory — facts are current and
authoritative there; a summary is prose and can lag). Summaries are scoped by
user_id/thread_id from the app, never from the model.

Incrementality: the active summary version is extended rather than rebuilt
from scratch — the prior summary text plus the newly-old turns are re-folded,
so a new version is O(new turns) in input, not O(thread).
"""

from __future__ import annotations

import psycopg

SUMMARIZE_PROMPT = (
    "You compress a loan-copilot conversation into a compact running summary.\n"
    "Rules:\n"
    "- Preserve every FACT and NUMBER stated or decided: income, debts, rates, "
    "LTV, lock terms, FICO, dates, and commitments (\"I will ...\").\n"
    "- NEVER compute or derive a number. Only transcribe values that are "
    "actually stated. No arithmetic, no DTI/payment/LTV math.\n"
    "- The CURRENT FACTS block lists the user's authoritative current values. "
    "When the conversation states an older value that the current facts "
    "supersede, record the CURRENT value and note the change; never re-assert "
    "a superseded value as current.\n"
    "- Do not invent anything not present in the material.\n"
    "- End each bullet with the turn sequence numbers it came from, like "
    "\"(turns 1-4)\".\n"
    "- Output plain-text bullets, no preamble, no headings."
)


def _est_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def fold_text(chat, material: str, *, max_tokens: int = 500) -> str:
    """One LLM call turning older conversation into a running summary."""
    out = chat.complete(
        [
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {"role": "user", "content": material},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return out.text


def read_active(conn: psycopg.Connection, user_id: str, thread_id: str) -> dict | None:
    """The latest summary version for a thread, if any."""
    cur = conn.execute(
        """
        SELECT summary_id, version, source_turn_ids, covers_through_seq, text, tokens
        FROM mem.summary
        WHERE user_id = %s AND thread_id = %s
        ORDER BY version DESC LIMIT 1
        """,
        (user_id, thread_id),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "summary_id": row[0],
        "version": row[1],
        "source_turn_ids": row[2],
        "covers_through_seq": row[3],
        "text": row[4],
        "tokens": row[5],
    }


def _turns(conn: psycopg.Connection, user_id: str, thread_id: str) -> list[dict]:
    cur = conn.execute(
        """
        SELECT turn_id, seq, role, content FROM mem.turn
        WHERE user_id = %s AND thread_id = %s
        ORDER BY seq
        """,
        (user_id, thread_id),
    )
    return [
        {"turn_id": r[0], "seq": r[1], "role": r[2], "content": r[3]}
        for r in cur.fetchall()
    ]


def summarize_older(
    conn: psycopg.Connection,
    *,
    user_id: str,
    thread_id: str,
    chat=None,
    summarizer=None,
    keep_k: int = 10,
    current_facts: str | None = None,
) -> dict | None:
    """Fold turns outside the verbatim window into the running summary.

    Returns the active summary dict, or None when the thread is still within
    the verbatim window (nothing to summarize) or the summary is already
    current. `summarizer` is callable(material_str) -> str; when omitted the
    LLM `chat` client is used, so tests can inject a deterministic one.
    `current_facts` (when given) is the user's authoritative fact block, so
    the summary never re-asserts a superseded conversation value as current.
    """
    turns = _turns(conn, user_id, thread_id)
    if len(turns) <= keep_k:
        return None
    fold = turns[: len(turns) - keep_k]   # oldest rows, oldest first
    boundary = fold[-1]["seq"]
    active = read_active(conn, user_id, thread_id)
    if active is not None and active["covers_through_seq"] == boundary:
        return active

    covered = active["covers_through_seq"] if active else 0
    new_rows = [t for t in turns if covered < t["seq"] <= boundary]
    parts: list[str] = []
    if current_facts:
        parts.append(f"[CURRENT FACTS — authoritative; superseded "
                     f"conversation values must yield to these]\n{current_facts}")
    if active:
        parts.append(f"[summary so far]\n{active['text']}")
    parts.append("\n".join(f"{t['role']}: {t['content']}" for t in new_rows))
    material = "\n\n".join(parts)

    if summarizer is not None:
        text = summarizer(material)
    else:
        if chat is None:
            raise ValueError("summarize_older needs chat or a summarizer")
        text = fold_text(chat, material)

    ids = list(active["source_turn_ids"] if active else [])
    ids += [t["turn_id"] for t in new_rows]
    version = (active["version"] + 1) if active else 1
    cur = conn.execute(
        """
        INSERT INTO mem.summary
            (user_id, thread_id, version, source_turn_ids,
             covers_through_seq, text, tokens)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING summary_id
        """,
        (user_id, thread_id, version, ids, boundary, text, _est_tokens(text)),
    )
    summary_id = int(cur.fetchone()[0])
    return {
        "summary_id": summary_id,
        "version": version,
        "source_turn_ids": ids,
        "covers_through_seq": boundary,
        "text": text,
        "tokens": _est_tokens(text),
    }


def format_summary(s: dict) -> str:
    return f"## Conversation summary (through turn {s['covers_through_seq']})\n{s['text']}"
