"""Unit 3 — entity memory.

What it stores: durable facts about nouns in the application — the applicant,
co-borrower, property, loan. Facts are APPEND-ONLY: a new value supersedes the
old one in one transaction and keeps the full history, so a corrected income
never silently rewrites the record and an auditor can always see what the
agent believed, and when.

When to use it: values that outlive the conversation and are needed across
threads (income, employment length, property units, loan amount).
What it is NOT for: conversation content (episodic), policy text (knowledge
base), how the agent should behave (persona).

Scope: user_id and thread_id come from the app. thread_id on a fact records
WHERE the fact was learned, not where it may be read — a fact learned in one
thread is visible to later threads; that is what makes it entity memory.
"""

from __future__ import annotations

import json

import psycopg

KINDS = ("applicant", "coborrower", "property", "loan")

ALLOWED_ATTRS: dict[str, tuple[str, ...]] = {
    "applicant": (
        "income_monthly", "income_annual", "employment_months",
        "fico", "dti", "first_time_buyer",
    ),
    "coborrower": ("income_monthly", "income_annual", "employment_months", "fico"),
    "property": ("units", "purchase_price", "property_type", "address"),
    "loan": ("loan_amount", "down_payment", "ltv", "purpose"),
}

EXTRACTION_PROMPT = """Extract factual statements about loan parties from the message below.

Allowed kinds: applicant, coborrower, property, loan.
Attributes per kind:
- applicant and coborrower: income_monthly, income_annual, employment_months, fico, dti, first_time_buyer
- property: units, purchase_price, property_type, address
- loan: loan_amount, down_payment, ltv, purpose

Rules:
- Extract ONLY facts explicitly stated in the message; transcribe numbers exactly as heard.
- "84,000" -> 84000, "84k" -> 84000, "75 percent" -> 75, "two years" -> employment_months 24.
- Income stated per month -> income_monthly; per year or annually -> income_annual.
- Skip questions, requests, and anything unknown.
- If no eligible fact exists, return an empty list.

Return JSON exactly in this shape: {{"facts": [{{"kind": "...", "name": "...", "attr": "...", "value": <string|number|boolean>}}]}}

Examples:
message: "Jane makes 105,000 a year now" -> {{"facts": [{{"kind": "applicant", "name": "Jane", "attr": "income_annual", "value": 105000}}]}}
message: "what is the max LTV for a two-unit investment" -> {{"facts": []}}
message: "the property at 48 Elm Street has 2 units" -> {{"facts": [{{"kind": "property", "name": "48 Elm Street", "attr": "units", "value": 2}}]}}

Message to process:"""


def get_or_create_entity(
    conn: psycopg.Connection,
    *,
    user_id: str,
    kind: str,
    name: str,
) -> int:
    if kind not in KINDS:
        raise ValueError(f"unknown entity kind: {kind}")
    cur = conn.execute(
        """
        INSERT INTO mem.entity (user_id, kind, name)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, kind, name) DO UPDATE SET user_id = EXCLUDED.user_id
        RETURNING entity_id
        """,
        (user_id, kind, name),
    )
    return int(cur.fetchone()[0])


def active_fact(
    conn: psycopg.Connection, entity_id: int, attr: str
) -> dict | None:
    cur = conn.execute(
        """
        SELECT fact_id, value, user_id, thread_id FROM mem.fact
        WHERE entity_id = %s AND attr = %s AND status = 'active'
        """,
        (entity_id, attr),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {"fact_id": row[0], "value": row[1],
            "user_id": row[2], "thread_id": row[3]}


def write_fact(
    conn: psycopg.Connection,
    *,
    user_id: str,
    thread_id: str,
    kind: str,
    name: str,
    attr: str,
    value,
    source_kind: str = "stated",
    source_ref: str | None = None,
    skip_if_same: bool = True,
) -> dict:
    """Insert a new fact version. Supersedes the active one atomically.

    Same-value writes are no-ops when skip_if_same (idempotence for retries).
    Returns {"fact_id", "supersedes_id", "changed"}.
    """
    if attr not in ALLOWED_ATTRS.get(kind, ()):
        raise ValueError(f"attr '{attr}' not allowed for kind '{kind}'")

    entity_id = get_or_create_entity(conn, user_id=user_id, kind=kind, name=name)
    with conn.transaction():  # savepoint if nested, txn if not — nesting-safe
        cur = conn.execute(
            "SELECT fact_id, value FROM mem.fact "
            "WHERE entity_id = %s AND attr = %s AND status = 'active' FOR UPDATE",
            (entity_id, attr),
        )
        row = cur.fetchone()
        active_fact_id, old_value = (row[0], row[1]) if row else (None, None)
        if old_value is not None and skip_if_same and old_value == value:
            return {"fact_id": active_fact_id, "supersedes_id": active_fact_id,
                    "changed": False}
        if active_fact_id is not None:
            conn.execute(
                "UPDATE mem.fact SET status = 'superseded' WHERE fact_id = %s",
                (active_fact_id,),
            )
        cur = conn.execute(
            """
            INSERT INTO mem.fact
                (entity_id, user_id, thread_id, attr, value,
                 supersedes_id, source_kind, source_ref)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING fact_id
            """,
            (entity_id, user_id, thread_id, attr, _dumps(value),
             active_fact_id, source_kind, source_ref),
        )
        fact_id = int(cur.fetchone()[0])
    return {"fact_id": fact_id, "supersedes_id": active_fact_id, "changed": True}


def _dumps(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def current_facts(conn: psycopg.Connection, entity_id: int) -> dict:
    """All active facts for one entity: attr -> value."""
    cur = conn.execute(
        """
        SELECT attr, value FROM mem.fact
        WHERE entity_id = %s AND status = 'active'
        """,
        (entity_id,),
    )
    return {a: v for a, v in cur.fetchall()}


def fact_history(conn: psycopg.Connection, entity_id: int, attr: str) -> list[dict]:
    """The full version trail of one fact, oldest first."""
    cur = conn.execute(
        """
        SELECT fact_id, value, status, supersedes_id, source_kind, created_at
        FROM mem.fact
        WHERE entity_id = %s AND attr = %s
        ORDER BY fact_id
        """,
        (entity_id, attr),
    )
    return [
        {"fact_id": r[0], "value": r[1], "status": r[2],
         "supersedes_id": r[3], "source_kind": r[4], "created_at": r[5]}
        for r in cur.fetchall()
    ]


def resolve(
    conn: psycopg.Connection,
    *,
    user_id: str,
    kind: str,
    name: str,
) -> dict | None:
    """Entity + current facts in one read — the agent's answer source."""
    cur = conn.execute(
        """
        SELECT entity_id FROM mem.entity
        WHERE user_id = %s AND kind = %s AND name = %s
        """,
        (user_id, kind, name),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {"entity_id": row[0], "facts": current_facts(conn, row[0])}
