# Security model — fact history protection

This documents exactly what the fact-history protections enforce, and what they
do **not**. The short version: they are enforced against the **application write
path**, not tamper-proof against a privileged SQL role.

## What is enforced

Three protections land in `infra/migrations/009_fact_immutable.sql` (plus the
audit trail from `008_audit.sql`):

1. `mem.fact_audit` is immutable — a BEFORE UPDATE/DELETE trigger always raises.
2. `mem.fact` is value-append-only — a BEFORE UPDATE trigger allows only the
   `active -> superseded` status flip and rejects any other column change.
3. `mem.fact` DELETE is blocked unless a session flag is set inside the
   transaction (`SET LOCAL mem.allow_fact_delete = 'on'`).

The only application path that sets the flag is `entity.purge_user()`, whose
callers are the eval runner, demo scripts and test cleanup — never the agent
write path, and the FastAPI app exposes no delete/reset route.

**Net effect:** ordinary application code cannot silently rewrite or delete fact
history; every supersession is an audited append.

## What is NOT enforced (be precise about this)

- **The flag is not authority.** `mem.allow_fact_delete` is an ordinary
  two-part custom setting. Stock PostgreSQL accepts `SET` of a two-part
  parameter name from any role and offers **no `REVOKE` path** for such a
  placeholder (the `SET` privilege added in PG 15 exists to *grant* access to
  superuser-only parameters, not to restrict custom placeholders). Anyone who
  can issue SQL and holds `DELETE` on `mem.fact` can set the flag and delete.
- **The dev compose runs the app as a superuser.** `docker-compose.yml` sets
  `POSTGRES_USER=acme`, and in the official Postgres image that role is a
  superuser. A superuser (or the table owner) can drop or disable the triggers.
- **`TRUNCATE` does not fire row triggers.** The protections are row triggers, so
  a `TRUNCATE mem.fact` bypasses them entirely.
- **`mem.entity` is neither protected nor audited.** `purge_user` deletes entity
  rows too; after a purge the `fact_audit` rows' `entity_id` no longer joins to a
  name.

So the honest framing is: **enforced against the application write path, not
tamper-proof against a privileged SQL role.** This is the intended scope for a
portfolio/dev project — it protects against bugs and unintended code paths, not
against a role that can already rewrite the database.

## Production hardening (what would make it stronger)

1. Run the application as a **non-superuser, non-owner** role.
2. `REVOKE DELETE, TRUNCATE ON mem.fact` and
   `REVOKE DELETE, UPDATE, TRUNCATE ON mem.fact_audit` from the app role.
3. `GRANT UPDATE (status) ON mem.fact` (column-level) to back the
   value-append-only rule at the privilege layer too.
4. Give `DELETE` on `mem.fact` to a **separate maintenance role** used only by
   demo/test resets — or replace the GUC check in the trigger with a
   `pg_has_role(current_user, '<maintenance role>', 'member')` test, so the gate
   checks role membership rather than a setting any role can flip.
5. Consider auditing `mem.entity` if entity identity is part of the record you
   need to protect.
6. Be aware that row triggers never fire on `TRUNCATE`; if resets use `TRUNCATE`,
   the trigger is not the thing preventing a silent wipe.
