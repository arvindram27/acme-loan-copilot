"""Unit 5 — toolbox: tool choice + tool log.

Tool CHOICE is vector search over tool descriptions, so the planner picks
among tools by meaning rather than by regex on the user's words.
Tool LOG remembers what ran: identical (tool, canonical args) within the
tool's TTL is a cache hit — reused, not re-executed — labeled 'cache' in the
outcome so the answer (and the eval) can tell reuse from fresh compute.
Large results are offloaded to a file under var/tool_artifacts/; the row
stores pointer + preview, and load_result() reassembles the full value.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import psycopg

from agentmem.settings import PROJECT_ROOT
from agentmem.tools import TOOLS

OFFLOAD_BYTES = 8 * 1024
ARTIFACT_DIR = PROJECT_ROOT / "var" / "tool_artifacts"

PLANNER_PROMPT = """You are the tool-planning layer of a loan copilot.
Given the context summary below, decide the next step.

Tools available:
{menu}

Rules:
- Compute nothing yourself. If a tool matches the need, call it with exactly the arguments it needs.
- MANDATORY CALLS, even when the arithmetic looks trivial or the numbers are already in the question: any affordability/payment/DTI question uses affordability_calculator; any rate/lock question uses rate_lookup; any LTV/price/loan-size cross-check uses ltv_check.
- affordability_calculator applies Acme's 45 percent policy DTI itself — do NOT pass max_dti_pct.
- If a "required_tool" is named in the context, you MUST emit its tool action FIRST, with arguments read from the question. Do not answer instead of calling a required tool.
- Numeric facts must come from tool outputs, never from your own arithmetic.
- If no tool helps, or you already have what you need to answer, answer now.
- Respond ONLY with JSON: {{"action": "tool", "name": "...", "arguments": {{...}}}} or {{"action": "answer"}}"""


def canonical_args(args: dict) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def args_hash(tool_name: str, args: dict) -> str:
    return hashlib.sha256(f"{tool_name}|{canonical_args(args)}"
                          .encode()).hexdigest()[:32]


def sync_tools(conn: psycopg.Connection, embedder) -> int:
    """Upsert the registry (descriptions + embeddings). Idempotent."""
    n = 0
    for name, spec in TOOLS.items():
        conn.execute(
            """
            INSERT INTO mem.tool (name, description, params_hint, ttl_default)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
              SET description = EXCLUDED.description,
                  params_hint = EXCLUDED.params_hint,
                  ttl_default = EXCLUDED.ttl_default
            """,
            (name, spec["description"], json.dumps(spec["params"]),
             spec["ttl"]),
        )
        row = conn.execute(
            "SELECT tool_id, description, embedding IS NULL "
            "FROM mem.tool WHERE name = %s",
            (name,),
        ).fetchone()
        tool_id, desc, no_embedding = int(row[0]), row[1], row[2]
        needs_embed = no_embedding or desc != spec["description"]
        if needs_embed:
            conn.execute(
                "UPDATE mem.tool SET embedding = %s::text::vector "
                "WHERE tool_id = %s",
                (_vec(embedder.embed(name + " " + spec["description"])),
                 tool_id),
            )
            n += 1
    return n


def _vec(embedding: list[float]) -> str:
    return "[" + ",".join(f"{x:.9g}" for x in embedding) + "]"


def search_tools(
    conn: psycopg.Connection, query_embedding: list[float], k: int = 3
) -> list[dict]:
    cur = conn.execute(
        """
        SELECT tool_id, name, description, params_hint, ttl_default,
               1 - (embedding <=> %s::vector) AS similarity
        FROM mem.tool WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector LIMIT %s
        """,
        (_vec(query_embedding), _vec(query_embedding), k),
    )
    return [
        {"tool_id": r[0], "name": r[1], "description": r[2],
         "params": r[3], "ttl": r[4], "similarity": float(r[5])}
        for r in cur.fetchall()
    ]


def find_cached(
    conn: psycopg.Connection, tool_id: int, h: str, ttl: int
) -> dict | None:
    """Latest completed call for this (tool, args) that is still fresh."""
    if ttl <= 0:
        return None
    cur = conn.execute(
        """
        SELECT call_id, result, result_bytes, artifact_path, computed_at
        FROM mem.tool_call
        WHERE tool_id = %s AND args_hash = %s AND source = 'fresh'
          AND computed_at + make_interval(secs => %s) > now()
        ORDER BY computed_at DESC LIMIT 1
        """,
        (tool_id, h, ttl),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "call_id": row[0], "result": row[1], "bytes": row[2],
        "artifact_path": row[3], "computed_at": row[4], "row": row,
    }


def load_result(
    conn: psycopg.Connection, row: tuple, artifact_dir: Path = ARTIFACT_DIR
) -> dict:
    """Reassemble a full result from a tool_call row (artifact or inline)."""
    artifact_path = row[3]
    if artifact_path:
        return json.loads((artifact_dir / artifact_path).read_text("utf-8"))
    return row[1]


def run_tool(
    conn: psycopg.Connection,
    *,
    user_id: str,
    thread_id: str,
    name: str,
    args: dict,
) -> dict:
    """Cache-check, execute, log and offload one tool call."""
    spec = TOOLS.get(name)
    if spec is None:
        raise KeyError(f"unknown tool: {name}")
    cur = conn.execute(
        "SELECT tool_id, ttl_default FROM mem.tool WHERE name = %s", (name,)
    )
    row = cur.fetchone()
    if row is None:  # registry not synced
        conn.execute(
            """
            INSERT INTO mem.tool (name, description, params_hint, ttl_default)
            VALUES (%s, %s, %s, %s)
            """,
            (name, spec["description"], json.dumps(spec["params"]), spec["ttl"]),
        )
        row = conn.execute(
            "SELECT tool_id, ttl_default FROM mem.tool WHERE name = %s",
            (name,),
        ).fetchone()
    tool_id, ttl = int(row[0]), int(row[1])
    h = args_hash(name, args)

    cached = find_cached(conn, tool_id, h, ttl)
    if cached:
        full = load_result(conn, cached["row"])
        return {
            "name": name, "args": args, "call_id": cached["call_id"],
            "source": "cache", "result_for_model": _model_result(full, None),
            "result_full": full,
            "result_bytes": cached["bytes"], "offloaded": False,
        }

    try:
        result = spec["fn"](args, conn)
        data = json.dumps(result, sort_keys=True)
        nbytes = len(data.encode())
        artifact_path = None
        if nbytes > OFFLOAD_BYTES:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            artifact_path = f"{h}.json"
            (ARTIFACT_DIR / artifact_path).write_text(data, "utf-8")
        stored = (
            {"offloaded": True, "bytes": nbytes,
             "preview": result.get("full_text", "")[:400]}
            if artifact_path else result
        )  # pointer + preview only; the artifact file carries the payload
        cur = conn.execute(
            """
            INSERT INTO mem.tool_call
                (user_id, thread_id, tool_id, args_hash, args, source,
                 result, result_bytes, artifact_path)
            VALUES (%s, %s, %s, %s, %s, 'fresh', %s, %s, %s)
            RETURNING call_id
            """,
            (
                user_id, thread_id, tool_id, h,
                json.dumps(args), json.dumps(stored), nbytes,
                artifact_path,
            ),
        )
        call_id = int(cur.fetchone()[0])
        return {
            "name": name, "args": args, "call_id": call_id,
            "source": "fresh",
            "result_for_model": _model_result(result, artifact_path),
            "result_full": result,
            "result_bytes": nbytes,
            "offloaded": bool(artifact_path),
        }
    except (ValueError, KeyError, TypeError) as exc:
        conn.execute(
            """
            INSERT INTO mem.tool_call
                (user_id, thread_id, tool_id, args_hash, args, source, error)
            VALUES (%s, %s, %s, %s, %s, 'error', %s)
            """,
            (user_id, thread_id, tool_id, h, json.dumps(args), str(exc)),
        )
        raise


def hub(result) -> int:
    return len(json.dumps(result).encode())


def _model_result(result: dict, artifact_path: str | None) -> str:
    """What the model sees: full small results, preview for offloaded ones."""
    text = json.dumps(result, indent=None)
    if artifact_path is None and len(text) <= OFFLOAD_BYTES:
        return text
    preview = result.get("full_text", "")[:400]
    return json.dumps({
        "offloaded": True,
        "artifact": artifact_path,
        "bytes": hub(result),
        "preview": preview + " ... [offloaded; read the artifact for the rest]",
        # the model is told the artifact exists but must not pretend to have
        # read it: results shown are what the preview contains
    })


def required_tool(text: str) -> str | None:
    """Deterministic backstop for the planner's weakest habit: small models
    like to compute in their heads. Scenario categories that MUST go through
    a tool are matched by keyword, and the planner is told it is required.
    The eval later verifies the call actually happened, not just the answer.

    Recall questions are exempt: asking what was SAID is answered from
    memory, not by recomputing (recomputing with fresh, possibly-wrong args
    is exactly how a recall answer gets poisoned)."""
    t = text.lower()
    if any(k in t for k in (
        "earlier", "remind me", "recall", "which one was it",
        "mentioned", "what did we say", "do you remember",
    )):
        return None
    if "ltv" in t or "loan-to-value" in t:
        return "ltv_check"
    if any(k in t for k in (
        "payment", "afford", "affordability", "qualify", "dti",
        "how much can", "debt",
    )):
        return "affordability_calculator"
    if any(k in t for k in ("rate", "lock", "interest", "apr")):
        return "rate_lookup"
    return None


def planner_payload(*, question: str, facts: dict, results_so_far: list) -> str:
    """The JSON context the planner reads. required_tool is a hard
    instruction, not a suggestion; see required_tool()."""
    return json.dumps({
        "question": question,
        "known_facts": facts,
        "results_so_far": results_so_far,
        "required_tool": required_tool(question),
    })


def run_tool_loop(
    conn,
    *,
    user_id: str,
    thread_id: str,
    chat,
    messages: list[dict],
    text: str,
    query_vec,
    facts: dict,
    use_tools: bool = True,
    tool_rounds: int = 3,
) -> list[dict]:
    """Plan/run tool calls until the planner says answer or rounds run out.

    Shared by both backends: the toolbox is deterministic application logic,
    not the memory store being contrasted. Rules: the planner computes nothing
    itself; tool results arrive as user-role blocks labeled fresh/cache; each
    result is logged to mem.tool_call with its args hash so reuse is real.
    """
    outcomes: list[dict] = []
    seen_hashes: set[str] = set()
    menu = ""
    if use_tools:
        found = search_tools(conn, query_vec, k=4)
        menu = "\n".join(
            f"[{t['name']}] {t['description']} "
            f"params: {json.dumps(t['params'])}"
            for t in found
        )
        # Ranked tools first; the full registry (with exact param names)
        # always follows so the planner can never plan argument-blind.
        menu += "\n" + "\n".join(
            f"[{n}] params: {json.dumps(s['params'])} ttl={s['ttl']}s"
            for n, s in TOOLS.items()
        )
    rounds = 0
    while rounds < tool_rounds:
        plan_out = chat.complete(
            [
                {"role": "system",
                 "content": PLANNER_PROMPT.format(menu=menu)},
                {"role": "user",
                 "content": planner_payload(
                     question=text,
                     facts=facts,
                     results_so_far=[
                         {"tool": t["name"],
                          "result": json.loads(t["result_for_model"])
                          if t["result_for_model"].startswith("{")
                          else t["result_for_model"]}
                         for t in outcomes
                     ],
                 )},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        try:
            decision = json.loads(plan_out.text)
        except json.JSONDecodeError:
            break
        name = decision.get("name")
        if decision.get("action") != "tool" or name not in TOOLS:
            break
        args = decision.get("arguments") or {}
        h = args_hash(name, args)
        if h in seen_hashes:
            # The planner asked for an identical call this turn; the
            # result is already in context. Re-computing teaches the
            # planner nothing and doubles cost for nothing.
            messages.append({
                "role": "user",
                "content": (f"## Duplicate request\n{name} was already "
                            "called with exactly these arguments this "
                            "turn; its result is above. Respond with "
                            '{"action": "answer"} now.'),
            })
            rounds += 1
            continue
        seen_hashes.add(h)
        try:
            t = run_tool(
                conn, user_id=user_id, thread_id=thread_id,
                name=name, args=args,
            )
            outcomes.append(t)
            note = (f"## Tool result — {t['name']} ({t['source']})\n"
                    f"{t['result_for_model']}\n\n"
                    "Use this result. Call another tool or give the "
                    "final answer. If any policy rule, fee, or limit is "
                    "involved, cite it as [title §chunk] from the "
                    "retrieved excerpts.")
        except (ValueError, KeyError, TypeError) as exc:
            note = (f"## Tool error — {decision.get('name')}\n{exc}\n"
                    "Fix the arguments or answer without the tool.")
        messages.append({"role": "user", "content": note})
        rounds += 1
    return outcomes


def recent_calls(
    conn: psycopg.Connection, *, user_id: str, limit: int = 10
) -> list[dict]:
    cur = conn.execute(
        """
        SELECT t.name, c.source, c.result_bytes, c.artifact_path, c.created_at
        FROM mem.tool_call c JOIN mem.tool t ON t.tool_id = c.tool_id
        WHERE c.user_id = %s ORDER BY c.created_at DESC LIMIT %s
        """,
        (user_id, limit),
    )
    return [
        {"tool": r[0], "source": r[1], "bytes": r[2],
         "artifact": r[3], "created_at": r[4]}
        for r in cur.fetchall()
    ]
