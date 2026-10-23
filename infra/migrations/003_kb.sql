-- 003_kb.sql — Unit 4: knowledge base (the copilot's shared policy corpus).
-- Deliberately NOT user-scoped: the corpus belongs to the app, not to a
-- user. The scope rule (user_id/thread_id from the app) applies to the
-- user-memory tables: turn, persona, entity/fact.

CREATE TABLE IF NOT EXISTS mem.doc (
    doc_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title      text NOT NULL,
    version    text NOT NULL,
    source_url text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (title, version)
);

CREATE TABLE IF NOT EXISTS mem.chunk (
    chunk_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_id     bigint NOT NULL REFERENCES mem.doc (doc_id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    text       text NOT NULL,
    text_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding  vector(1536),
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunk_tsv_idx ON mem.chunk USING gin (text_tsv);
CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw
    ON mem.chunk USING hnsw (embedding vector_cosine_ops);
