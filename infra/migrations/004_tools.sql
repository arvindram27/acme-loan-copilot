-- 004_tools.sql — Unit 5: toolbox + tool log.
-- The tool REGISTRY is descriptive; tool CHOICE is vector search over
-- descriptions. The tool LOG is the memory of what was run: keyed by args
-- hash so identical calls can be reused while fresh, with large results
-- offloaded to an artifact (the row keeps a pointer and a preview).

CREATE TABLE IF NOT EXISTS mem.tool (
    tool_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    description text NOT NULL,
    params_hint jsonb NOT NULL,
    ttl_default int NOT NULL DEFAULT 0,  -- seconds; 0 = always recompute
    embedding   vector(1536)
);

CREATE TABLE IF NOT EXISTS mem.tool_call (
    call_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id       text NOT NULL,
    thread_id     text NOT NULL,
    tool_id       bigint NOT NULL REFERENCES mem.tool (tool_id),
    args_hash     text NOT NULL,
    args          jsonb NOT NULL,
    source        text NOT NULL CHECK (source IN ('fresh', 'cache', 'error')),
    result        jsonb,
    result_bytes  int,
    artifact_path text,
    error         text,
    computed_at   timestamptz NOT NULL DEFAULT now(),
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tool_call_fresh_idx
    ON mem.tool_call (tool_id, args_hash, computed_at DESC);
CREATE INDEX IF NOT EXISTS tool_call_scope_idx
    ON mem.tool_call (user_id, thread_id);
