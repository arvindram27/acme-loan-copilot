"""Unit 4 — knowledge base with hybrid retrieval.

What it stores: the copilot's reference corpus — Acme underwriting policy,
product sheets. Chunked paragraphs, embedded at 1536 dims, full-text indexed
for keyword search.

When to use it: answers that must cite policy ("what's the max LTV for …").
What it is NOT for: anything the user said (episodic/entity memory). Policy
text is shared across users — the scope rule does not apply here by design,
and every chunk row is traceable to (doc, chunk_index).

Read paths: keyword (tsvector), vector (cosine), hybrid (reciprocal rank
fusion over both). Retrieval A/B (recall@k, MRR) is measured later by the
evaluation harness.
"""

from __future__ import annotations

import psycopg

CHUNK_TARGET = 900  # approx chars per chunk
RRF_K = 60


def _vec_param(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.9g}" for x in embedding) + "]"


def _chunk_paragraphs(text: str) -> list[str]:
    """Paragraph boundaries first; oversized paragraphs split on word
    boundaries (crude but deterministic — no mid-chunk sentence fiddling)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for p in paragraphs:
        if len(p) <= CHUNK_TARGET:
            chunks.append(p)
            continue
        buf = ""
        for word in p.split():
            if buf and len(buf) + 1 + len(word) > CHUNK_TARGET:
                chunks.append(buf)
                buf = word
            else:
                buf = f"{buf} {word}".strip()
        if buf:
            chunks.append(buf)
    return chunks


def add_doc(
    conn: psycopg.Connection,
    *,
    title: str,
    version: str,
    text: str,
    embedding_fn,
    source_url: str | None = None,
) -> dict:
    """Insert or replace a doc's chunks. Idempotent per (title, version)."""
    chunks = _chunk_paragraphs(text)
    with conn.transaction():
        cur = conn.execute(
            """
            INSERT INTO mem.doc (title, version, source_url)
            VALUES (%s, %s, %s)
            ON CONFLICT (title, version)
            DO UPDATE SET source_url = EXCLUDED.source_url
            RETURNING doc_id
            """,
            (title, version, source_url),
        )
        doc_id = int(cur.fetchone()[0])
        conn.execute("DELETE FROM mem.chunk WHERE doc_id = %s", (doc_id,))
        for i, chunk_text in enumerate(chunks):
            conn.execute(
                """
                INSERT INTO mem.chunk (doc_id, chunk_index, text, embedding)
                VALUES (%s, %s, %s, %s::text::vector)
                """,
                (doc_id, i, chunk_text, _vec_param(embedding_fn(chunk_text))),
            )
    return {"doc_id": doc_id, "chunks": len(chunks)}


def keyword_search(
    conn: psycopg.Connection, query: str, k: int = 10
) -> list[dict]:
    cur = conn.execute(
        """
        SELECT c.chunk_id, d.title, c.chunk_index, c.text,
               ts_rank(c.text_tsv, websearch_to_tsquery('english', %s)) AS score
        FROM mem.chunk c JOIN mem.doc d ON d.doc_id = c.doc_id
        WHERE c.text_tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC LIMIT %s
        """,
        (query, query, k),
    )
    return [_row(r, by="keyword") for r in cur.fetchall()]


def vector_search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    k: int = 10,
    min_score: float | None = None,
) -> list[dict]:
    vec = _vec_param(query_embedding)
    where = ""
    params: list = [vec]
    if min_score is not None:
        where = " HAVING 1 - (c.embedding <=> %s::vector) >= %s"
        params.extend([vec, min_score])
    cur = conn.execute(
        f"""
        SELECT c.chunk_id, d.title, c.chunk_index, c.text,
               1 - (c.embedding <=> %s::vector) AS score
        FROM mem.chunk c JOIN mem.doc d ON d.doc_id = c.doc_id
        WHERE c.embedding IS NOT NULL
        GROUP BY c.chunk_id, d.title, c.chunk_index, c.text, c.embedding{where}
        ORDER BY score DESC LIMIT {k}
        """,
        params,
    )
    return [_row(r, by="vector") for r in cur.fetchall()]


def hybrid_search(
    conn: psycopg.Connection,
    *,
    query: str,
    query_embedding: list[float],
    k: int = 5,
    min_score: float | None = 0.0,
) -> list[dict]:
    """Reciprocal rank fusion of keyword and vector rankings.

    The vector side enforces min_score so unrelated chunks cannot ride the
    fusion purely on rank position.
    """
    cur = conn.execute(
        """
        WITH kw AS (
            SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY score DESC) AS rnk
            FROM (
                SELECT c.chunk_id AS chunk_id,
                       ts_rank(c.text_tsv,
                               websearch_to_tsquery('english', %s)) AS score
                FROM mem.chunk c
                WHERE c.text_tsv @@ websearch_to_tsquery('english', %s)
                LIMIT 20
            ) s
        ),
        vec AS (
            SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY score DESC) AS rnk
            FROM (
                SELECT c.chunk_id AS chunk_id,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM mem.chunk c
                WHERE c.embedding IS NOT NULL
                  AND 1 - (c.embedding <=> %s::vector) >= %s
                LIMIT 20
            ) s
        )
        SELECT c.chunk_id, d.title, c.chunk_index, c.text,
               COALESCE(1.0 / (%s + kw.rnk), 0) + COALESCE(1.0 / (%s + vec.rnk), 0) AS rrf
        FROM mem.chunk c
             JOIN mem.doc d ON d.doc_id = c.doc_id
             LEFT JOIN kw ON kw.chunk_id = c.chunk_id
             LEFT JOIN vec ON vec.chunk_id = c.chunk_id
        WHERE kw.rnk IS NOT NULL OR vec.rnk IS NOT NULL
        ORDER BY rrf DESC LIMIT %s
        """,
        (query, query, _vec_param(query_embedding), _vec_param(query_embedding),
         min_score, RRF_K, RRF_K, k),
    )
    return [_row(r, by="hybrid", score="rrf") for r in cur.fetchall()]


def _row(r, by: str, score: str = "score") -> dict:
    out = {
        "chunk_id": r[0], "title": r[1], "chunk_index": r[2],
        "text": r[3], by: r[4], "by": by,
    }
    return out


def format_chunks(chunks: list[dict]) -> str:
    """Prompt-ready excerpts with provenance labels."""
    return "\n\n".join(
        f"[{c['title']} §{c['chunk_index']}] {c['text']}" for c in chunks
    )
