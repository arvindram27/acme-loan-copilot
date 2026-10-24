"""Unit 6 — semantic answer cache: memoize a whole answer by (scope, epoch, q).

A repeat or paraphrase of an earlier question can be answered from cache with
~0 completion tokens. A hit requires THREE things to agree:
  scope   same user_id + thread_id
  epoch   the user's fact-epoch unchanged (any later fact write bumps it, so
          a cached answer can never reflect stale facts)
  meaning query embedding within `min_score` of the stored question

Because the epoch guard is the safety mechanism, cache invalidation is
implicit: no DELETE job, no TTL race — a new fact simply makes old entries
unreachable for that user. `expires_at` is a backstop for questions whose
answer could age even when facts have not (e.g. "what is today's rate"),
which the caller can leave unset for stable recall questions.

What it is NOT for: turns that must go through a tool (the eval enforces
tool discipline; serving a cached affordability answer would skip the tool),
and turns that state NEW facts (a cached answer would skip extraction).
`is_question` gates both out.
"""

from __future__ import annotations

import psycopg

_QUESTION_WORDS = (
    "what", "who", "which", "when", "where", "how", "why",
    "can", "could", "do", "does", "is", "are", "would", "will",
    "remind", "tell", "repeat", "summarize",
)


def _vec(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.9g}" for x in embedding) + "]"


def is_question(text: str) -> bool:
    """Heuristic gate: only interrogative/imperative turns enter or hit the
    cache. A bare fact statement ("I make 9,000 now") must never be served
    from a cached answer, because serving it would skip fact extraction."""
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    return any(t.startswith(w) or f" {w} " in f" {t} " for w in _QUESTION_WORDS)


def fact_epoch(conn: psycopg.Connection, user_id: str) -> int:
    """The user's fact generation: max(fact_id). Advances on every fact write,
    so it is a monotonic invalidation token for the whole user's cache."""
    cur = conn.execute(
        "SELECT COALESCE(MAX(fact_id), 0) FROM mem.fact WHERE user_id = %s",
        (user_id,),
    )
    return int(cur.fetchone()[0])


def lookup(
    conn: psycopg.Connection,
    *,
    user_id: str,
    thread_id: str,
    query_embedding: list[float],
    min_score: float = 0.62,
    epoch: int | None = None,
) -> dict | None:
    """Nearest cached answer for this (scope, epoch). None on miss or expiry.

    min_score is tuned to text-embedding-3-small: genuine paraphrases of a
    short question sit around 0.67-0.74, unrelated questions around <=0.53,
    so 0.62 separates them. The scope + epoch guards are the safety net."""
    epoch = fact_epoch(conn, user_id) if epoch is None else epoch
    vec = _vec(query_embedding)
    cur = conn.execute(
        """
        SELECT cache_id, question, answer,
               1 - (embedding <=> %s::vector) AS similarity
        FROM mem.cache
        WHERE user_id = %s AND thread_id = %s AND epoch = %s
          AND (expires_at IS NULL OR expires_at > now())
        ORDER BY embedding <=> %s::vector
        LIMIT 1
        """,
        (vec, user_id, thread_id, epoch, vec),
    )
    row = cur.fetchone()
    if row is None or float(row[3]) < min_score:
        return None
    return {
        "cache_id": row[0],
        "question": row[1],
        "answer": row[2],
        "similarity": float(row[3]),
    }


def store(
    conn: psycopg.Connection,
    *,
    user_id: str,
    thread_id: str,
    question: str,
    query_embedding: list[float],
    answer: str,
    epoch: int | None = None,
    ttl_seconds: int | None = None,
    expires_at=None,
) -> int:
    """Record an answer for future paraphrase hits. Returns cache_id."""
    epoch = fact_epoch(conn, user_id) if epoch is None else epoch
    if expires_at is None and ttl_seconds is not None:
        expires_at = f"now() + interval '{ttl_seconds} seconds'"
    cur = conn.execute(
        """
        INSERT INTO mem.cache
            (user_id, thread_id, epoch, question, answer, embedding, expires_at)
        VALUES (%s, %s, %s, %s, %s, %s::text::vector,
                %s::timestamptz)
        RETURNING cache_id
        """,
        (
            user_id, thread_id, epoch, question, answer,
            _vec(query_embedding),
            expires_at if expires_at is not None else None,
        ),
    )
    return int(cur.fetchone()[0])
