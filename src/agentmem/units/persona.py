"""Unit 2 — persona.

What it stores: who the agent should be for this user — style, strictness,
policies they insist on. Versioned: an update supersedes the active row and
inserts a new version in one transaction; exactly one active row per user is
enforced by a partial unique index in the schema.

When to use it: constant, thread-independent instructions that should shape
every request for a user.
What it is NOT for: facts that change per application (entity memory) or
conversation content (episodic memory). A persona row is *judgement*, typed
and versioned, not chat history.

change_trigger records why the version exists: app_default (installed by the
app), user_request ("no small talk", verbatim), implication (observed, not
spoken).
"""

from __future__ import annotations

import json

import psycopg

DEFAULT_PERSONA = {
    "name": "Acme Loan Copilot",
    "style": "concise, no small talk",
    "rules": [
        "Quote policy numbers only from the knowledge base",
        "Never invent rates; use tool results or say you don't know",
        "Flag any question you cannot answer with policy evidence",
    ],
}


def read_active(conn: psycopg.Connection, user_id: str) -> dict | None:
    cur = conn.execute(
        """
        SELECT persona_id, version, change_trigger, body
        FROM mem.persona
        WHERE user_id = %s AND status = 'active'
        """,
        (user_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    persona_id, version, trigger, body = row
    return {
        "persona_id": persona_id,
        "version": version,
        "change_trigger": trigger,
        "body": body,
    }


def read_version(conn: psycopg.Connection, user_id: str, version: int) -> dict | None:
    cur = conn.execute(
        """
        SELECT persona_id, status, change_trigger, body
        FROM mem.persona WHERE user_id = %s AND version = %s
        """,
        (user_id, version),
    )
    row = cur.fetchone()
    if row is None:
        return None
    persona_id, status, trigger, body = row
    return {
        "persona_id": persona_id,
        "status": status,
        "change_trigger": trigger,
        "body": body,
    }


def ensure_default(
    conn: psycopg.Connection,
    user_id: str,
    body: dict | None = None,
) -> dict:
    """Install the default persona if none exists; idempotent."""
    existing = read_active(conn, user_id)
    if existing is not None:
        return existing
    return update(conn, user_id, body or DEFAULT_PERSONA,
                  change_trigger="app_default")


def update(
    conn: psycopg.Connection,
    user_id: str,
    body: dict,
    *,
    change_trigger: str = "user_request",
) -> dict:
    """Supersede the active version and insert a new one in one unit of work.

    Uses `with conn.transaction()` (a savepoint when an outer transaction
    exists) instead of a manual commit, so it participates in the caller's
    unit of work rather than committing it early. See docs/architecture.md.
    """
    with conn.transaction():
        cur = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM mem.persona WHERE user_id = %s",
            (user_id,),
        )
        version = int(cur.fetchone()[0]) + 1
        conn.execute(
            """
            UPDATE mem.persona
            SET status = 'superseded'
            WHERE user_id = %s AND status = 'active'
            """,
            (user_id,),
        )
        cur = conn.execute(
            """
            INSERT INTO mem.persona (user_id, version, status, change_trigger, body)
            VALUES (%s, %s, 'active', %s, %s)
            RETURNING persona_id, version, change_trigger, body
            """,
            (user_id, version, change_trigger, json.dumps(body)),
        )
        persona_id, new_version, trigger, new_body = cur.fetchone()
    return {
        "persona_id": persona_id,
        "version": new_version,
        "change_trigger": trigger,
        "body": new_body,
    }


def format_persona(persona: dict | None) -> str:
    """Prompt-ready rendering. None stays None — the caller decides to ask."""
    if persona is None:
        return ""
    rules = "\n".join(f"- {r}" for r in persona["body"].get("rules", []))
    return (
        f"Persona (version {persona['version']}, {persona['change_trigger']}): "
        f"{persona['body'].get('style', '')}\n{rules}"
    )
