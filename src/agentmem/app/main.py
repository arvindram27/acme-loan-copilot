"""Acme Home Loans copilot — the "memory ladder" app (milestone 8 + redesign).

Pages: Overview (the ladder), Chat (with a live "The memory" panel and a
scratch<->langgraph switcher), one page per memory rung, Measure (evals), and a
data explorer. Everything is server-rendered Jinja2; the only JS is the bottom
data drawer.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agentmem.app.deps import (
    get_chat,
    get_embedder,
    get_pool,
    has_api_key,
    make_agent,
)
from agentmem.settings import get_settings

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Acme Home Loans — Memory Ladder")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_settings = get_settings()

# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------

RUNGS = [
    {"n": 0, "slug": "baseline", "title": "Stateless baseline", "pill": None,
     "table": None,
     "lede": "No memory at all. Every request starts from nothing.",
     "stores": "Nothing.",
     "when": "It is the control arm, not a feature.",
     "notfor": "Anything real.",
     "demo": "What is my current monthly income?",
     "demo_note": "The baseline answers \"I don't know your applicant\" — it has no memory."},
    {"n": 1, "slug": "conversation", "title": "Conversation", "pill": "short-term",
     "table": "mem.turn",
     "lede": "One row per turn, scoped by user and thread, with an embedding so a new question can recall old similar turns.",
     "stores": "What the agent and user said in this thread.",
     "when": "Anything said in this thread.",
     "notfor": "Durable facts or policy knowledge.",
     "demo": "What did we say my debts were earlier in this conversation?",
     "demo_note": "Recall is answered from the buffer and semantic recall, not recomputed."},
    {"n": 2, "slug": "persona", "title": "Persona", "pill": "session",
     "table": "mem.persona",
     "lede": "Who the agent should be for this user — style, strictness, rules they insist on. Versioned; one active version.",
     "stores": "Constant, thread-independent instructions.",
     "when": "Shaping every request for this user.",
     "notfor": "Facts that change per application, or chat content.",
     "demo": None,
     "demo_note": "Persona is read on every turn, but is not something you ask about — it is who answers."},
    {"n": 3, "slug": "entity", "title": "Entity facts", "pill": "semantic",
     "table": "mem.entity · mem.fact",
     "lede": "Durable facts about nouns — applicant, co-borrower, property, loan. Append-only: a correction supersedes, never rewrites.",
     "stores": "Values that outlive the conversation and span threads.",
     "when": "Anything that must stay current across threads.",
     "notfor": "Conversation content, policy text, or behavior.",
     "demo": "My raise came through — I make 8,750 a month now. What's my income on file?",
     "demo_note": "The old value is superseded, not deleted — both survive, the current one wins."},
    {"n": 4, "slug": "knowledge", "title": "Knowledge (RAG)", "pill": "semantic",
     "table": "mem.doc · mem.chunk",
     "lede": "The underwriting corpus, chunked and embedded. Hybrid keyword+vector retrieval with reciprocal rank fusion.",
     "stores": "Policy text, shared across users by design.",
     "when": "Answers that must cite policy.",
     "notfor": "Anything the user said.",
     "demo": "What is Acme's maximum LTV for a 2-unit investment property?",
     "demo_note": "The answer cites [title §chunk]; the number comes from retrieved text only."},
    {"n": 5, "slug": "toolbox", "title": "Toolbox", "pill": "episodic",
     "table": "mem.tool · mem.tool_call",
     "lede": "Deterministic tools that keep arithmetic in code. Tool choice is semantic search; the log reuses fresh results by args-hash.",
     "stores": "What ran, and its result, with a TTL.",
     "when": "Any computation — never let the model do arithmetic.",
     "notfor": "Storing facts; the log is provenance, not truth.",
     "demo": "What is the most I could pay each month under Acme's DTI limit?",
     "demo_note": "The planner calls affordability_calculator; the number is the tool's, not the model's."},
    {"n": 6, "slug": "working", "title": "Working memory", "pill": "episodic",
     "table": "mem.summary",
     "lede": "When a thread outgrows the verbatim window, older turns fold into a compressed summary that cites the turns it covers.",
     "stores": "Compressed history of older turns.",
     "when": "A long interview needs to stay inside the token budget.",
     "notfor": "Durable facts — entity memory is authoritative; a summary is prose.",
     "demo": "Can you summarize everything you know about my finances so far?",
     "demo_note": "Watch the summary appear in the memory panel once the thread is long enough."},
    {"n": 7, "slug": "cache", "title": "Semantic cache", "pill": "procedural",
     "table": "mem.cache",
     "lede": "Memoizes a whole answer by scope + fact-epoch + embedding. A repeat or paraphrase is served with ~0 completion tokens.",
     "stores": "Answerable questions and their answers.",
     "when": "A repeat or paraphrase of an already-answered question.",
     "notfor": "Turns that must go through a tool, or that state new facts.",
     "demo": "Sorry, one more time — what is my current income per month?",
     "demo_note": "The second ask is served from cache — 0 completion tokens."},
]

# rung 8 is "Measure" and lives at /evals


def _db():
    return get_pool().connection()


def _history(conn, user_id: str, thread_id: str, backend: str) -> list[dict]:
    rows = conn.execute(
        "SELECT seq, question, answer FROM mem.app_trace "
        "WHERE user_id = %s AND thread_id = %s AND backend = %s ORDER BY seq",
        (user_id, thread_id, backend),
    ).fetchall()
    out: list[dict] = []
    for _seq, question, answer in rows:
        out.append({"role": "user", "content": question})
        out.append({"role": "assistant", "content": answer})
    return out


def _threads(conn, user_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT thread_id FROM mem.turn WHERE user_id = %s ORDER BY 1",
        (user_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _drawer_stats(conn) -> dict:
    tables = [r[0] for r in conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'mem' "
        "ORDER BY tablename").fetchall()]
    if not tables:
        return {"tables": 0, "rows": 0}
    parts = " + ".join(f"(SELECT count(*) FROM mem.{t})" for t in tables)
    total = conn.execute(f"SELECT {parts}").fetchone()[0]
    return {"tables": len(tables), "rows": int(total or 0)}


def _global_ctx(request):
    drawer = {"tables": 0, "rows": 0}
    try:
        with _db() as conn:
            drawer = _drawer_stats(conn)
    except Exception:
        pass
    return {"rungs": RUNGS, "drawer": drawer, "model": _settings.chat_model}


TEMPLATES.context_processors.append(_global_ctx)


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def overview(request: Request):
    return TEMPLATES.TemplateResponse(request, "overview.html", {
        "active": "overview",
    })


# ---------------------------------------------------------------------------
# chat + memory panel
# ---------------------------------------------------------------------------

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request, user_id: str = "maya", thread_id: str = "a",
              backend: str = "scratch"):
    backend = backend if backend in ("scratch", "langgraph") else "scratch"
    with _db() as conn:
        threads = _threads(conn, user_id)
        history = _history(conn, user_id, thread_id, backend)
        traces = _trace_rows(conn, user_id, thread_id, backend)
    panel = _build_panel(None, traces, backend)
    return TEMPLATES.TemplateResponse(request, "chat.html", {
        "active": "chat", "user_id": user_id, "thread_id": thread_id,
        "backend": backend, "threads": threads, "history": history,
        "panel": panel, "last": None,
        "error": None if has_api_key() else "MEM_API_KEY is not set — chat is disabled.",
    })


def _trace_dict(last) -> dict:
    return {
        "prompt_tokens": last.prompt_tokens,
        "context_tokens": last.context_tokens,
        "tokens": last.tokens,
        "recent_turns": last.recent_turns,
        "recalled_turns": last.recalled_turns,
        "summary_used": last.summary_used,
        "summary_turns": last.summary_turns,
        "answer_cache_hit": last.answer_cache_hit,
        "facts_used": last.facts_used,
        "retrieved_refs": [r.get("title") for r in last.retrieved_refs],
        "tools": [t.get("name") for t in last.tool_calls],
        "tool_sources": [t.get("source") for t in last.tool_calls],
    }


def _trace_rows(conn, user_id, thread_id, backend, limit: int = 12):
    rows = conn.execute(
        "SELECT seq, question, trace FROM mem.app_trace "
        "WHERE user_id = %s AND thread_id = %s AND backend = %s "
        "ORDER BY seq DESC LIMIT %s",
        (user_id, thread_id, backend, limit),
    ).fetchall()
    out = []
    for seq, q, trace in reversed(rows):
        t = trace if isinstance(trace, dict) else json.loads(trace)
        out.append({"seq": seq, "question": q, **t})
    return out


MAX_CONTEXT_TOKENS = 2800


def _build_panel(last, traces: list[dict], backend: str) -> dict:
    latest = _trace_dict(last) if last is not None else (traces[-1] if traces else None)
    chart = [
        {"seq": t.get("seq"), "pt": t.get("prompt_tokens", 0),
         "cache": bool(t.get("answer_cache_hit"))}
        for t in traces
    ]
    if latest is not None and latest.get("answer_cache_hit") and chart:
        chart[-1]["cache"] = True
    return {
        "latest": latest, "chart": chart, "backend": backend,
        "max_tokens": MAX_CONTEXT_TOKENS,
    }


@app.post("/chat", response_class=HTMLResponse)
def chat_post(
    request: Request,
    user_id: str = Form("maya"),
    thread_id: str = Form("a"),
    message: str = Form(...),
    backend: str = Form("scratch"),
):
    backend = backend if backend in ("scratch", "langgraph") else "scratch"
    last = None
    error = None
    if not has_api_key():
        error = "MEM_API_KEY is not set — chat is disabled."
    elif not message.strip():
        error = "Type a message first."
    else:
        with _db() as conn:
            agent = make_agent(backend, user_id, conn)
            last = agent.answer(thread_id=thread_id, text=message)
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM mem.app_trace "
                "WHERE user_id = %s AND thread_id = %s AND backend = %s",
                (user_id, thread_id, backend),
            ).fetchone()[0]
            trace = _trace_dict(last)
            conn.execute(
                "INSERT INTO mem.app_trace "
                "(user_id, thread_id, backend, seq, question, answer, trace) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user_id, thread_id, backend, seq, message, last.text,
                 json.dumps(trace, default=str)),
            )
    with _db() as conn:
        threads = _threads(conn, user_id)
        history = _history(conn, user_id, thread_id, backend)
        traces = _trace_rows(conn, user_id, thread_id, backend)
    panel = _build_panel(last, traces, backend)
    return TEMPLATES.TemplateResponse(request, "chat.html", {
        "active": "chat", "user_id": user_id, "thread_id": thread_id,
        "backend": backend, "threads": threads, "history": history,
        "panel": panel, "last": last, "error": error,
    })


# ---------------------------------------------------------------------------
# rung pages
# ---------------------------------------------------------------------------

@app.get("/rung/{n}", response_class=HTMLResponse)
def rung_page(request: Request, n: int):
    if n == 8:
        return RedirectResponse(url="/evals")
    if not (0 <= n < 8):
        return TEMPLATES.TemplateResponse(request, "rung.html", {
            "active": None, "rung": None, "live": None, "user_id": "maya",
        }, status_code=404)
    rung = RUNGS[n]
    with _db() as conn:
        live = _rung_live(conn, "maya", n)
    return TEMPLATES.TemplateResponse(request, "rung.html", {
        "active": f"rung-{n}", "rung": rung, "live": live, "user_id": "maya",
    })


def _rung_live(conn, user_id: str, n: int) -> dict:
    if n == 1:
        rows = conn.execute(
            "SELECT thread_id, seq, role, content FROM mem.turn "
            "WHERE user_id = %s ORDER BY thread_id, seq DESC LIMIT 30",
            (user_id,),
        ).fetchall()
        return {"kind": "turns", "rows": [list(r) for r in reversed(rows)]}
    if n == 2:
        row = conn.execute(
            "SELECT version, change_trigger, body FROM mem.persona "
            "WHERE user_id = %s AND status = 'active'", (user_id,),
        ).fetchone()
        return {"kind": "persona", "persona": row}
    if n == 3:
        rows = conn.execute(
            "SELECT e.kind, e.name, f.attr, f.value, f.status FROM mem.fact f "
            "JOIN mem.entity e ON e.entity_id = f.entity_id "
            "WHERE f.user_id = %s ORDER BY e.kind, e.name, f.attr, f.fact_id",
            (user_id,),
        ).fetchall()
        return {"kind": "facts", "rows": [list(r) for r in rows]}
    if n == 4:
        rows = conn.execute(
            "SELECT d.title, c.chunk_index, left(c.text, 180) FROM mem.chunk c "
            "JOIN mem.doc d ON d.doc_id = c.doc_id ORDER BY d.doc_id, c.chunk_index LIMIT 12",
        ).fetchall()
        return {"kind": "chunks", "rows": [list(r) for r in rows]}
    if n == 5:
        rows = conn.execute(
            "SELECT t.name, c.source, c.args, c.result_bytes FROM mem.tool_call c "
            "JOIN mem.tool t ON t.tool_id = c.tool_id "
            "WHERE c.user_id = %s ORDER BY c.created_at DESC LIMIT 15",
            (user_id,),
        ).fetchall()
        return {"kind": "tools", "rows": [list(r) for r in rows]}
    if n == 6:
        rows = conn.execute(
            "SELECT thread_id, version, covers_through_seq, left(text, 240) "
            "FROM mem.summary WHERE user_id = %s ORDER BY thread_id, version DESC LIMIT 15",
            (user_id,),
        ).fetchall()
        return {"kind": "summaries", "rows": [list(r) for r in rows]}
    if n == 7:
        rows = conn.execute(
            "SELECT thread_id, left(question, 60), left(answer, 60), epoch "
            "FROM mem.cache WHERE user_id = %s ORDER BY created_at DESC LIMIT 15",
            (user_id,),
        ).fetchall()
        return {"kind": "cache", "rows": [list(r) for r in rows]}
    return {"kind": "none"}


# ---------------------------------------------------------------------------
# measure (evals)
# ---------------------------------------------------------------------------

@app.get("/evals", response_class=HTMLResponse)
def evals_page(request: Request):
    with _db() as conn:
        runs = conn.execute(
            "SELECT run_id, backend, model, workload, n_items, metrics, "
            " jev_cost_usd, started_at, finished_at, commit_hash, workload_hash "
            "FROM mem.eval_run ORDER BY run_id DESC LIMIT 50"
        ).fetchall()
    return TEMPLATES.TemplateResponse(request, "evals.html", {
        "active": "measure", "runs": runs,
    })


@app.get("/evals/{run_id}", response_class=HTMLResponse)
def eval_run_page(request: Request, run_id: int):
    with _db() as conn:
        run = conn.execute(
            "SELECT run_id, backend, model, workload, n_items, metrics, "
            " jev_cost_usd, started_at, finished_at, commit_hash, workload_hash "
            "FROM mem.eval_run WHERE run_id = %s", (run_id,),
        ).fetchone()
        items = conn.execute(
            "SELECT item_id, item_no, user_id, thread_id, question, expected, "
            " outcome FROM mem.eval_item WHERE run_id = %s ORDER BY item_no",
            (run_id,),
        ).fetchall()
        grades = conn.execute(
            "SELECT i.item_no, g.grader, g.check_name, g.pass, g.score, "
            " g.probability, g.escalated, g.rationale FROM mem.grade g "
            "JOIN mem.eval_item i ON i.item_id = g.item_id "
            "WHERE i.run_id = %s ORDER BY i.item_no, g.grader, g.check_name",
            (run_id,),
        ).fetchall()
    if run is None:
        return TEMPLATES.TemplateResponse(request, "eval_run.html", {
            "active": "measure", "run": None, "items": [], "grades": [],
        }, status_code=404)

    grade_map: dict[int, list] = {}
    for g in grades:
        grade_map.setdefault(g[0], []).append(g)

    items_out = []
    for it in items:
        expected = it[5] if isinstance(it[5], dict) else json.loads(it[5])
        outcome = it[6] if isinstance(it[6], dict) else json.loads(it[6])
        items_out.append({
            "item_no": it[1], "user_id": it[2], "thread_id": it[3],
            "question": it[4],
            "answer": (outcome or {}).get("text", ""),
            "expected": json.dumps(expected, indent=2),
            "grades": grade_map.get(it[1], []),
        })
    return TEMPLATES.TemplateResponse(request, "eval_run.html", {
        "active": "measure", "run": run, "items": items_out,
    })


# ---------------------------------------------------------------------------
# data explorer + consolidated memory
# ---------------------------------------------------------------------------

@app.get("/data", response_class=HTMLResponse)
def data_page(request: Request, table: str | None = None, limit: int = 50):
    limit = max(1, min(limit, 200))
    with _db() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'mem' "
            "ORDER BY tablename").fetchall()]
        cols: list[str] = []
        rows = []
        if table and table in tables:
            cols = [r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'mem' AND table_name = %s "
                "ORDER BY ordinal_position", (table,)).fetchall()]
            rows = conn.execute(
                f"SELECT * FROM mem.{table} ORDER BY 1 DESC LIMIT %s", (limit,),
            ).fetchall()
    return TEMPLATES.TemplateResponse(request, "data.html", {
        "active": "data", "tables": tables, "table": table,
        "cols": cols, "rows": rows, "limit": limit,
    })


@app.get("/memory", response_class=HTMLResponse)
def memory_page(request: Request, user_id: str = "maya"):
    with _db() as conn:
        persona_row = conn.execute(
            "SELECT version, change_trigger, body FROM mem.persona "
            "WHERE user_id = %s AND status = 'active'", (user_id,)).fetchone()
        facts = conn.execute(
            "SELECT e.kind, e.name, f.attr, f.value, f.status FROM mem.fact f "
            "JOIN mem.entity e ON e.entity_id = f.entity_id "
            "WHERE f.user_id = %s ORDER BY e.kind, e.name, f.attr, f.fact_id",
            (user_id,)).fetchall()
        turns = conn.execute(
            "SELECT thread_id, seq, role, content FROM mem.turn "
            "WHERE user_id = %s ORDER BY thread_id, seq", (user_id,)).fetchall()
        summaries = conn.execute(
            "SELECT thread_id, version, covers_through_seq, text FROM mem.summary "
            "WHERE user_id = %s ORDER BY thread_id, version", (user_id,)).fetchall()
        cache_rows = conn.execute(
            "SELECT thread_id, question, answer, created_at FROM mem.cache "
            "WHERE user_id = %s ORDER BY created_at DESC LIMIT 20", (user_id,)).fetchall()
    return TEMPLATES.TemplateResponse(request, "memory.html", {
        "active": "memory", "user_id": user_id, "persona": persona_row,
        "facts": facts, "turns": turns, "summaries": summaries,
        "cache_rows": cache_rows,
    })
