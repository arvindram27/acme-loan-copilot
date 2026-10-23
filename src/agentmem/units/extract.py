"""LLM fact extraction, guarded by deterministic validation.

The model proposes facts (kind, name, attr, value) from what the user said;
code rejects anything outside the allowlist, coerces numbers, and only
validated facts reach entity.write_fact. A wrong extracted number flows into
every later step — so extraction is only trusted after validation, exactly
like the course's rule: check extracted data with deterministic rules.
"""

from __future__ import annotations

import json
import re

from agentmem.providers.chat import ChatClient
from agentmem.units.entity import ALLOWED_ATTRS, EXTRACTION_PROMPT, KINDS

NUMERIC_ATTRS = {
    "income_monthly", "income_annual", "employment_months", "fico",
    "dti", "units", "purchase_price", "loan_amount", "down_payment", "ltv",
}
_BOOL_ATTRS = {"first_time_buyer"}

_INT_PATTERN = re.compile(r"^-?\d+\s*$")


def build_message() -> str:
    return EXTRACTION_PROMPT


def extract_facts(chat: ChatClient, text: str) -> tuple[list[dict], str]:
    """Model proposes facts. Returns (raw_facts, model_used). Raises nothing.

    Withhold judgement here: validation happens in validate_facts, which is
    testable without the model.
    """
    out = chat.complete(
        [
            {"role": "system", "content": build_message()},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(out.text)
    except json.JSONDecodeError:
        return [], out.model
    return list(data.get("facts", [])), out.model


def _coerce_value(attr: str, raw) -> object | None:
    if attr in NUMERIC_ATTRS:
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            return raw
        cleaned = str(raw).strip().replace("$", "").replace(",", "").rstrip("%.")
        try:
            return int(cleaned) if _INT_PATTERN.match(cleaned) else float(cleaned)
        except (ValueError, TypeError):
            return None
    if attr in _BOOL_ATTRS:
        if isinstance(raw, bool):
            return raw
        low = str(raw).strip().lower()
        return {"true": True, "yes": True, "false": False, "no": False}.get(low)
    text = str(raw).strip()
    return text or None


def validate_facts(raw_facts: list[dict]) -> tuple[list[dict], list[str]]:
    """Filter model output against the allowlist. Returns (valid, dropped).

    A valid fact: kind in KINDS, name non-empty short string, attr in that
    kind's allowlist, value coercible. Notes why each fact was dropped.
    """
    valid: list[dict] = []
    dropped: list[str] = []
    for f in raw_facts:
        if not isinstance(f, dict):
            dropped.append(f"not an object: {f!r}")
            continue
        kind, name, attr, value = (
            f.get("kind"), f.get("name"), f.get("attr"), f.get("value")
        )
        if kind not in KINDS:
            dropped.append(f"unknown kind: {kind!r}")
        elif not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            dropped.append(f"bad name for {attr!r}: {name!r}")
        elif attr not in ALLOWED_ATTRS.get(kind, ()):
            dropped.append(f"attr {attr!r} not allowed for kind {kind!r}")
        else:
            # "'Ravi Chandra's annual income'" would otherwise yield entity
            # name "Ravi Chandras" — a different person, silently. Strip it.
            cleaned_name = re.sub(r"['\u2019]s?$", "", name.strip())
            coerced = _coerce_value(attr, value)
            if coerced is None:
                dropped.append(f"value {value!r} not coercible for {attr!r}")
            else:
                valid.append({"kind": kind, "name": cleaned_name,
                              "attr": attr, "value": coerced})
    return valid, dropped


def write_validated(
    conn, *, user_id: str, thread_id: str, facts: list[dict]
) -> list[dict]:
    """Persist validated facts. Returns what was actually written.

    First-person facts (name "SELF", from the extraction prompt) are bound to
    the app-supplied user_id, so "I make 8,750" becomes the applicant entity
    named by user_id — the same entity later turns read and supersede."""
    written = []
    for f in facts:
        out = entity_write(conn, user_id=user_id, thread_id=thread_id, fact=f)
        out["attr"] = f["attr"]
        out["value"] = f["value"]
        written.append(out)
    return written


def entity_write(conn, *, user_id: str, thread_id: str, fact: dict) -> dict:
    # Local import avoids a cycle at module load.
    from agentmem.units import entity

    name = user_id if fact["name"] == "SELF" else fact["name"]
    return entity.write_fact(
        conn, user_id=user_id, thread_id=thread_id, kind=fact["kind"],
        name=name, attr=fact["attr"], value=fact["value"],
        source_kind="stated",
    )
