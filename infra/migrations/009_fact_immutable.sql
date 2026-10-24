-- 009_fact_immutable.sql — truly append-only fact history.
--
-- Review re-check item #8: history was audited but still mutable. This makes:
--   1. mem.fact_audit immutable (no UPDATE/DELETE);
--   2. mem.fact value-append-only (only the status active->superseded flip is
--      allowed; any other UPDATE raises);
--   3. mem.fact DELETE blocked unless the maintenance escape hatch is set
--      (SET LOCAL mem.allow_fact_delete = 'on' inside a transaction), used only
--      by entity.purge_user() for test/demo resets.

-- 1. Audit rows are append-only.
CREATE OR REPLACE FUNCTION mem.protect_fact_audit() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'mem.fact_audit is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS fact_audit_immutable ON mem.fact_audit;
CREATE TRIGGER fact_audit_immutable
    BEFORE UPDATE OR DELETE ON mem.fact_audit
    FOR EACH ROW EXECUTE FUNCTION mem.protect_fact_audit();

-- 2 + 3. mem.fact: value-append-only, delete-gated.
CREATE OR REPLACE FUNCTION mem.protect_fact() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        IF NEW.fact_id <> OLD.fact_id
           OR NEW.entity_id <> OLD.entity_id
           OR NEW.user_id IS DISTINCT FROM OLD.user_id
           OR NEW.thread_id IS DISTINCT FROM OLD.thread_id
           OR NEW.attr IS DISTINCT FROM OLD.attr
           OR NEW.value IS DISTINCT FROM OLD.value
           OR NEW.supersedes_id IS DISTINCT FROM OLD.supersedes_id
           OR NEW.source_kind IS DISTINCT FROM OLD.source_kind
           OR NEW.source_ref IS DISTINCT FROM OLD.source_ref THEN
            RAISE EXCEPTION 'mem.fact is append-only; only status may change';
        END IF;
        IF NOT (OLD.status = 'active' AND NEW.status = 'superseded') THEN
            RAISE EXCEPTION 'mem.fact status may only go active -> superseded';
        END IF;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        IF current_setting('mem.allow_fact_delete', true) IS DISTINCT FROM 'on' THEN
            RAISE EXCEPTION
                'mem.fact is append-only; DELETE needs SET LOCAL mem.allow_fact_delete = ''on''';
        END IF;
        RETURN OLD;
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS fact_protect ON mem.fact;
CREATE TRIGGER fact_protect
    BEFORE UPDATE OR DELETE ON mem.fact
    FOR EACH ROW EXECUTE FUNCTION mem.protect_fact();
