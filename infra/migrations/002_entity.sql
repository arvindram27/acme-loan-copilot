-- 002_entity.sql — Unit 3: entity memory.
-- Entities typed by kind; facts are append-only rows where a new value
-- SUPERSEDES the active one (never updates it). Exactly one active fact per
-- (entity, attr) is enforced by the partial unique index.

CREATE TABLE IF NOT EXISTS mem.entity (
    entity_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    text NOT NULL,
    kind       text NOT NULL CHECK (kind IN ('applicant', 'coborrower', 'property', 'loan')),
    name       text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, kind, name)
);

CREATE TABLE IF NOT EXISTS mem.fact (
    fact_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entity_id     bigint NOT NULL REFERENCES mem.entity (entity_id),
    user_id       text   NOT NULL,
    thread_id     text   NOT NULL,
    attr          text   NOT NULL,
    value         jsonb  NOT NULL,
    status        text   NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'superseded')),
    supersedes_id bigint REFERENCES mem.fact (fact_id),
    source_kind   text   NOT NULL DEFAULT 'stated'
                  CHECK (source_kind IN ('stated', 'tool', 'derived')),
    source_ref    text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS fact_entity_idx ON mem.fact (entity_id, attr);
CREATE INDEX IF NOT EXISTS fact_scope_idx ON mem.fact (user_id, thread_id);

-- One active fact per (entity, attr) — the supersession invariant.
CREATE UNIQUE INDEX IF NOT EXISTS fact_active_idx
    ON mem.fact (entity_id, attr) WHERE status = 'active';
