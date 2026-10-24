"""Lazily-initialized shared resources for the web app.

Everything is a module-level singleton created on first use, so importing the
app does not require the database or the API key to be reachable — pages that
only read the DB (memory/data/evals) work even without MEM_API_KEY.
"""

from __future__ import annotations

from psycopg_pool import ConnectionPool

from agentmem.db import migrate
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings

_settings = get_settings()

_pool: ConnectionPool | None = None
_chat: ChatClient | None = None
_embedder: Embedder | None = None


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
