-- 001_init.sql — foundation: pgvector, the mem schema, conversation and persona.
-- Later milestones add tables in their own migration files.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS mem;

-- ---------------------------------------------------------------------------
-- 1. Conversation (episodic memory)
-- One row per turn. Scoped by user_id/thread_id, seq is monotonic per thread.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mem.turn (
    turn_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    text NOT NULL,
    thread_id  text NOT NULL,
    seq        int  NOT NULL,
    role       text NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content    text NOT NULL,
    tokens     int,
    embedding  vector(1536),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (thread_id, seq)
);

CREATE INDEX IF NOT EXISTS turn_scope_idx
    ON mem.turn (user_id, thread_id, seq);

CREATE INDEX IF NOT EXISTS turn_embedding_hnsw
    ON mem.turn USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- 2. Persona
-- Versioned; exactly one active row per user (enforced by the partial unique
-- index). Updates write a new version and mark the old one superseded in one
-- transaction.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mem.persona (
    persona_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id        text NOT NULL,
    version        int  NOT NULL,
    status         text NOT NULL DEFAULT 'active'
                   CHECK (status IN ('active', 'superseded')),
    change_trigger text NOT NULL CHECK (change_trigger IN
                   ('app_default', 'user_request', 'implication')),
    body           jsonb NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS persona_active_idx
    ON mem.persona (user_id) WHERE status = 'active';

-- ---------------------------------------------------------------------------
-- Migration bookkeeping
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mem.migrations (
    migration_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name           text NOT NULL UNIQUE,
    applied_at     timestamptz NOT NULL DEFAULT now()
);
