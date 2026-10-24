"""FastAPI app: chat, memory inspector, data explorer, evals replay."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agentmem.agent import MemoryAgent
from agentmem.app.deps import get_chat, get_embedder, get_pool, has_api_key

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Acme Home Loans — Copilot")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _db():
    """One connection checked out of the pool for the duration of a request."""
    return get_pool().connection()


def _history(conn, user_id: str, thread_id: str, k: int = 40) -> list[dict]:
    rows = conn.execute(
        """
        SELECT seq, role, content FROM mem.turn
        WHERE user_id = %s AND thread_id = %s
        ORDER BY seq DESC LIMIT %s
        """,
        (user_id, thread_id, k),
    ).fetchall()
    return [{"role": r[1], "content": r[2]} for r in reversed(rows)]


def _threads(conn, user_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT thread_id FROM mem.turn WHERE user_id = %s ORDER BY 1",
        (user_id,),
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/chat")


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request, user_id: str = "maya", thread_id: str = "a"):
    with _db() as conn:
        history = _history(conn, user_id, thread_id)
        threads = _threads(conn, user_id)
    return TEMPLATES.TemplateResponse(request, "chat.html", {
        "request": request, "user_id": user_id, "thread_id": thread_id,
        "history": history, "threads": threads, "last": None,
        "error": None if has_api_key() else "MEM_API_KEY is not set — chat is disabled.",
    })


@app.post("/chat", response_class=HTMLResponse)
def chat_post(
    request: Request,
    user_id: str = Form("maya"),
    thread_id: str = Form("a"),
    message: str = Form(...),
):
    last = None
    error = None
    if not has_api_key():
        error = "MEM_API_KEY is not set — chat is disabled."
    elif not message.strip():
        error = "Type a message first."
    else:
        with _db() as conn:
            agent = MemoryAgent(conn, user_id=user_id, chat=get_chat(),
                                embedder=get_embedder())
            last = agent.answer(thread_id=thread_id, text=message)
    with _db() as conn:
        history = _history(conn, user_id, thread_id)
        threads = _threads(conn, user_id)
    return TEMPLATES.TemplateResponse(request, "chat.html", {
        "request": request, "user_id": user_id, "thread_id": thread_id,
        "history": history, "threads": threads, "last": last, "error": error,
    })


# ---------------------------------------------------------------------------
# memory inspector
# ---------------------------------------------------------------------------

@app.get("/memory", response_class=HTMLResponse)
def memory_page(request: Request, user_id: str = "maya"):
    with _db() as conn:
        persona_row = conn.execute(
            "SELECT version, change_trigger, body FROM mem.persona "
            "WHERE user_id = %s AND status = 'active'",
            (user_id,),
        ).fetchone()
        facts = conn.execute(
            """
            SELECT e.kind, e.name, f.attr, f.value
            FROM mem.fact f JOIN mem.entity e ON e.entity_id = f.entity_id
            WHERE f.user_id = %s AND f.status = 'active'
            ORDER BY e.kind, e.name, f.attr
            """,
            (user_id,),
        ).fetchall()
        turns = conn.execute(
            "SELECT thread_id, seq, role, content FROM mem.turn "
            "WHERE user_id = %s ORDER BY thread_id, seq",
            (user_id,),
        ).fetchall()
        summaries = conn.execute(
            "SELECT thread_id, version, covers_through_seq, text "
            "FROM mem.summary WHERE user_id = %s ORDER BY thread_id, version",
            (user_id,),
        ).fetchall()
        cache_rows = conn.execute(
            "SELECT thread_id, question, answer, created_at FROM mem.cache "
            "WHERE user_id = %s ORDER BY created_at DESC LIMIT 20",
            (user_id,),
        ).fetchall()
        tool_rows = conn.execute(
            """
            SELECT t.name, c.source, c.args, c.result_bytes, c.created_at
            FROM mem.tool_call c JOIN mem.tool t ON t.tool_id = c.tool_id
            WHERE c.user_id = %s ORDER BY c.created_at DESC LIMIT 20
            """,
            (user_id,),
        ).fetchall()
    return TEMPLATES.TemplateResponse(request, "memory.html", {
        "request": request,
        "user_id": user_id,
        "persona": persona_row,
        "facts": facts,
        "turns": turns,
        "summaries": summaries,
        "cache_rows": cache_rows,
        "tool_rows": tool_rows,
    })


# ---------------------------------------------------------------------------
# data explorer
# ---------------------------------------------------------------------------

@app.get("/data", response_class=HTMLResponse)
def data_page(request: Request, table: str | None = None, limit: int = 50):
    limit = max(1, min(limit, 200))
    with _db() as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'mem' "
            "ORDER BY tablename"
        ).fetchall()]
        cols: list[str] = []
        rows = []
        if table and table in tables:
            cols = [r[0] for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'mem' AND table_name = %s "
                "ORDER BY ordinal_position",
                (table,),
            ).fetchall()]
            rows = conn.execute(
                f"SELECT * FROM mem.{table} ORDER BY 1 DESC LIMIT %s", (limit,),
            ).fetchall()
    return TEMPLATES.TemplateResponse(request, "data.html", {
        "request": request, "tables": tables, "table": table,
        "cols": cols, "rows": rows, "limit": limit,
    })


# ---------------------------------------------------------------------------
# evals replay
# ---------------------------------------------------------------------------

@app.get("/evals", response_class=HTMLResponse)
def evals_page(request: Request):
    with _db() as conn:
        runs = conn.execute(
            "SELECT run_id, backend, model, workload, n_items, metrics, "
            " jev_cost_usd, started_at, finished_at FROM mem.eval_run "
            "ORDER BY run_id DESC LIMIT 50"
        ).fetchall()
    return TEMPLATES.TemplateResponse(request, "evals.html", {
        "request": request, "runs": runs,
    })


@app.get("/evals/{run_id}", response_class=HTMLResponse)
def eval_run_page(request: Request, run_id: int):
    with _db() as conn:
        run = conn.execute(
            "SELECT run_id, backend, model, workload, n_items, metrics, "
            " jev_cost_usd, started_at, finished_at FROM mem.eval_run "
            "WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        items = conn.execute(
            "SELECT item_id, item_no, user_id, thread_id, question, "
            " expected, outcome FROM mem.eval_item WHERE run_id = %s "
            "ORDER BY item_no",
            (run_id,),
        ).fetchall()
        grades = conn.execute(
            """
            SELECT i.item_no, g.grader, g.check_name, g.pass, g.score,
                   g.probability, g.escalated, g.rationale
            FROM mem.grade g JOIN mem.eval_item i ON i.item_id = g.item_id
            WHERE i.run_id = %s ORDER BY i.item_no, g.grader, g.check_name
            """,
            (run_id,),
        ).fetchall()
    if run is None:
        return TEMPLATES.TemplateResponse(request, "eval_run.html", {
            "request": request, "run": None, "items": [], "grades": [],
        }, status_code=404)

    # group grades by item_no for the template
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
        "request": request, "run": run, "items": items_out,
        "grades": grades,
    })
