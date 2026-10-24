-- 006_working.sql — Unit 6: working memory (summaries) + semantic cache.

-- Working memory: rolling summaries of a thread's OLDER turns. When a thread
-- grows past the verbatim recency window, the turns that fall out are folded
-- into a compressed summary (which records the turn seqs it covers), so a
-- long interview stays inside the context budget without losing facts.
-- Summaries are append-only versions; the active one is the latest.
CREATE TABLE IF NOT EXISTS mem.summary (
    summary_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id            text NOT NULL,
    thread_id          text NOT NULL,
    version            int NOT NULL,
    source_turn_ids    bigint[] NOT NULL,   -- mem.turn.turn_id folded in
    covers_through_seq int NOT NULL,        -- seq of the last folded turn row
    text               text NOT NULL,
    tokens             int,
    created_at         timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, thread_id, version)
);
CREATE INDEX IF NOT EXISTS summary_thread_idx
    ON mem.summary (user_id, thread_id, version DESC);

-- Semantic answer cache: memoizes a full answer keyed by (scope, fact epoch,
-- question embedding). A hit requires the SAME user+thread and the SAME fact
-- epoch, so any later fact write (which advances the epoch) invalidates the
-- whole user's cached answers. expires_at bounds staleness independently.
CREATE TABLE IF NOT EXISTS mem.cache (
    cache_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id    text NOT NULL,
    thread_id  text NOT NULL,
    epoch      bigint NOT NULL,             -- max(fact_id) for user at store time
    question   text NOT NULL,
    answer     text NOT NULL,
    embedding  vector(1536) NOT NULL,
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS cache_scope_idx
    ON mem.cache (user_id, thread_id, epoch);
