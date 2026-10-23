"""Embedding adapter.

Hosted embeddings through the same OpenAI-compatible endpoint as chat
(OpenRouter serves /embeddings). An in-memory LRU avoids re-embedding the
repeated short strings a memory system asks for (persona text, cache scope,
etc.). Cache is per-process; persistence is the store's business, not this
client's.
"""

from functools import lru_cache

from openai import OpenAI

from agentmem.settings import Settings, get_settings


class Embedder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = OpenAI(
            base_url=self.settings.openrouter_base_url,
            api_key=self.settings.api_key or "unset",
        )
        self._embed = lru_cache(maxsize=1024)(self._embed_uncached)
        self.calls = 0

    def _embed_uncached(self, text: str) -> tuple[float, ...]:
        resp = self._client.embeddings.create(
            model=self.settings.embed_model,
            input=text,
            dimensions=self.settings.embed_dims,
        )
        self.calls += 1
        return tuple(float(x) for x in resp.data[0].embedding)

    def embed(self, text: str) -> list[float]:
        """One string -> one vector."""
        return list(self._embed(text))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Many strings -> many vectors (deduplicated through the LRU)."""
        return [self.embed(t) for t in texts]
