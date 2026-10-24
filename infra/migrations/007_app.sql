-- 007_app.sql — app telemetry (NOT memory). The web app records one row per
-- answered chat turn so the "memory" panel can chart per-turn token usage and
-- replay what the copilot resent. The trace jsonb holds the graded TraceOutcome
-- fields; the memory system itself is untouched by this table.

CREATE TABLE IF NOT EXISTS mem.app_trace (
    trace_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    text NOT NULL,
    thread_id  text NOT NULL,
    backend    text NOT NULL DEFAULT 'scratch',
    seq        int,
    question   text NOT NULL,
    answer     text NOT NULL,
    trace      jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS app_trace_scope_idx
    ON mem.app_trace (user_id, thread_id, seq);
