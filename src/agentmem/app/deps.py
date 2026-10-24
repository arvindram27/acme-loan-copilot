"""Lazily-initialized shared resources for the web app.

Everything is a module-level singleton created on first use, so importing the
app does not require the database or the API key to be reachable — pages that
only read the DB (memory/data/evals) work even without MEM_API_KEY.

Also owns the backend factory: `make_agent("scratch"|"langgraph", ...)` returns
either a hand-rolled MemoryAgent or a LangGraphAgent over agentmem_lg, so the UI
can offer the scratch<->langgraph switcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

from agentmem.agent import MemoryAgent
from agentmem.backends.langgraph import (
    LangGraphAgent,
    build_checkpointer,
    build_store,
)
from agentmem.db import migrate
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import PROJECT_ROOT, get_settings
from agentmem.units import knowledge

_settings = get_settings()

_pool: ConnectionPool | None = None
_chat: ChatClient | None = None
_embedder: Embedder | None = None
_lg: dict | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        migrate(_settings.database_url)
        _pool = ConnectionPool(conninfo=_settings.database_url,
                               min_size=1, max_size=8, open=True)
    return _pool


def get_chat() -> ChatClient:
    global _chat
    if _chat is None:
        _chat = ChatClient()
    return _chat


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def has_api_key() -> bool:
    return bool(_settings.api_key)


def _seed_kb(conn, embedder) -> None:
    n = conn.execute("SELECT COUNT(*) FROM mem.chunk").fetchone()[0]
    if n:
        return
    corpus = json.loads(
        (PROJECT_ROOT / "infra" / "seed" / "corpus.json").read_text(encoding="utf-8")
    )
    for doc in corpus["docs"]:
        knowledge.add_doc(conn, title=doc["title"],
                          version=corpus["version"], text=doc["text"],
                          embedding_fn=embedder.embed)


def get_langgraph() -> dict:
    """LangGraph backend resources: dedicated pool for the store/checkpointer,
    a plain connection for the shared KB/tool SQL, and the seeded corpus."""
    global _lg
    if _lg is None:
        migrate(_settings.database_url_lg)
        conn = psycopg.connect(_settings.database_url_lg, autocommit=True)
        _seed_kb(conn, get_embedder())
        # LangGraph's setup() runs CREATE INDEX CONCURRENTLY, which cannot run
        # inside a transaction, so its pool connections must be autocommit.
        pool = ConnectionPool(conninfo=_settings.database_url_lg,
                              min_size=1, max_size=4, open=True,
                              kwargs={"autocommit": True})
        store = build_store(pool, get_embedder())
        check = build_checkpointer(pool)
        _lg = {"conn": conn, "store": store, "check": check}
    return _lg


def make_agent(backend: str, user_id: str, conn):
    """One memory agent for a request. backend is 'scratch' or 'langgraph'."""
    if backend == "langgraph":
        lg = get_langgraph()
        return LangGraphAgent(
            user_id=user_id, chat=get_chat(), embedder=get_embedder(),
            conn=lg["conn"], store=lg["store"], checkpointer=lg["check"],
        )
    return MemoryAgent(conn, user_id=user_id, chat=get_chat(),
                       embedder=get_embedder())
