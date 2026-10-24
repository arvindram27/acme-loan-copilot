"""Unit 1 — conversation (episodic memory).

What it stores: one row per turn, scoped by user_id/thread_id, with an
embedding of the content so a *new* question can recall similar *old* turns
even from far back in the thread.

When to use it: anything the agent or the user said in this thread.
What it is NOT for: durable user facts (that is persona/entity memory) or
policy knowledge (the knowledge base).

Scope comes from the app: user_id and thread_id are arguments here, never
model output.
"""

from __future__ import annotations

import json

import psycopg


def _vec_param(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.9g}" for x in embedding) + "]"


def next_seq(conn: psycopg.Connection, thread_id: str) -> int:
    cur = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM mem.turn WHERE thread_id = %s",
        (thread_id,),
    )
    return int(cur.fetchone()[0])


def append_turn(
    conn: psycopg.Connection,
    *,
    user_id: str,
    thread_id: str,
    role: str,
    content: str,
    seq: int | None = None,
    embedding: list[float] | None = None,
    tokens: int | None = None,
    context_snapshot: dict | None = None,
) -> dict:
    """Insert one turn. seq defaults to next in thread. Returns the stored row.

    `context_snapshot` records the memory state that informed an assistant
    turn (fact ids/values, summary id, retrieved chunk ids), so an answer can
    be replayed against exactly what it saw.
    """
    seq = seq if seq is not None else next_seq(conn, thread_id)
    cur = conn.execute(
        """
        INSERT INTO mem.turn
            (user_id, thread_id, seq, role, content, tokens, embedding,
             context_snapshot)
        VALUES (%s, %s, %s, %s, %s, %s, %s::text::vector, %s)
        RETURNING turn_id, seq, created_at
        """,
        (
            user_id,
            thread_id,
            seq,
            role,
            content,
            tokens,
            _vec_param(embedding) if embedding is not None else None,
            json.dumps(context_snapshot) if context_snapshot is not None else None,
        ),
    )
    turn_id, stored_seq, created_at = cur.fetchone()
    return {
        "turn_id": turn_id,
        "seq": stored_seq,
        "role": role,
        "content": content,
        "created_at": created_at,
    }


def recent_turns(conn: psycopg.Connection, thread_id: str, k: int = 10) -> list[dict]:
    """The k most recent turns, oldest first — the base conversation context."""
    cur = conn.execute(
        """
        SELECT seq, role, content FROM mem.turn
        WHERE thread_id = %s
        ORDER BY seq DESC LIMIT %s
        """,
        (thread_id, k),
    )
    rows = cur.fetchall()
    return [{"seq": s, "role": r, "content": c} for s, r, c in reversed(rows)]


def recall(
    conn: psycopg.Connection,
    *,
    user_id: str,
    thread_id: str,
    query_embedding: list[float],
    k: int = 5,
    before_seq: int | None = None,
) -> list[dict]:
    """Semantic recall of older turns within the thread.

    before_seq keeps the read path away from turns the recent window already
    contains, so context never duplicates.
    """
    where = "user_id = %s AND thread_id = %s AND embedding IS NOT NULL"
    inner: list = [user_id, thread_id]
    if before_seq is not None:
        where += " AND seq < %s"
        inner.append(before_seq)
    # placeholder order: SELECT's similarity vec, WHERE fields, ORDER vec, LIMIT
    vec = _vec_param(query_embedding)
    params = [vec, *inner, vec, k]
    cur = conn.execute(
        f"""
        SELECT seq, role, content,
               1 - (embedding <=> %s::vector) AS similarity
        FROM mem.turn WHERE {where}
        ORDER BY embedding <=> %s::vector LIMIT %s
        """,
        params,
    )
    return [
        {"seq": s, "role": r, "content": c, "similarity": float(sim)}
        for s, r, c, sim in cur.fetchall()
    ]


def format_turns(turns: list[dict]) -> str:
    """Prompt-ready rendering of turns in seq order."""
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)
