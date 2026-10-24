-- 008_audit.sql — run-to-code linkage, context snapshots, immutable fact trail.
--
-- Review items #7 and #8: a stored eval run must answer "which commit, which
-- prompts, which graders, which dataset"; an answer must be replayable against
-- the exact fact versions that informed it; and fact history must not be
-- silently mutable.

-- 1. Eval-run linkage -------------------------------------------------------
ALTER TABLE mem.eval_run ADD COLUMN IF NOT EXISTS commit_hash     text;
ALTER TABLE mem.eval_run ADD COLUMN IF NOT EXISTS prompt_version  text;
ALTER TABLE mem.eval_run ADD COLUMN IF NOT EXISTS grader_config   jsonb;
ALTER TABLE mem.eval_run ADD COLUMN IF NOT EXISTS workload_hash   text;

-- 2. Per-turn context snapshot (fact ids/values, summary, retrieved chunks) --
ALTER TABLE mem.turn ADD COLUMN IF NOT EXISTS context_snapshot jsonb;

-- 3. Fact audit: every insert/update/delete is recorded with old/new values. --
-- Fact history can therefore never change *silently*: the application may
-- append a new version (which flips the previous row to superseded), but the
-- audit row carries the exact before/after state, and a delete is recorded
-- rather than erased. (A hard DELETE/UPDATE revoke is left to deployment
-- roles; this works with the single dev role too.)
CREATE TABLE IF NOT EXISTS mem.fact_audit (
    audit_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fact_id    bigint NOT NULL,
    entity_id  bigint NOT NULL,
    op         text NOT NULL CHECK (op IN ('insert', 'update', 'delete')),
    old_row    jsonb,
    new_row    jsonb,
    changed_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS fact_audit_fact_idx ON mem.fact_audit (fact_id);

CREATE OR REPLACE FUNCTION mem.audit_fact() RETURNS trigger AS $$
BEGIN
    INSERT INTO mem.fact_audit (fact_id, entity_id, op, old_row, new_row)
    VALUES (
        COALESCE(NEW.fact_id, OLD.fact_id),
        COALESCE(NEW.entity_id, OLD.entity_id),
        lower(TG_OP),
        CASE WHEN TG_OP IN ('UPDATE', 'DELETE') THEN to_jsonb(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN to_jsonb(NEW) ELSE NULL END
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS fact_audit ON mem.fact;
CREATE TRIGGER fact_audit
    AFTER INSERT OR UPDATE OR DELETE ON mem.fact
    FOR EACH ROW EXECUTE FUNCTION mem.audit_fact();
