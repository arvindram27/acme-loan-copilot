"""Live model calls against the real provider. Skipped without a key."""

import pytest

from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.settings import get_settings

s = get_settings()
requires_key = pytest.mark.skipif(not s.has_api_key, reason="MEM_API_KEY not set")

# These make real (paid) model calls, so they are part of the `live` selection.
pytestmark = pytest.mark.live


@requires_key
def test_chat_roundtrip():
    client = ChatClient()
    out = client.complete(
        [{"role": "user", "content": "Reply with the single word: ready."}],
        max_tokens=16,
    )
    assert out.text.strip(), "expected non-empty reply"
    assert out.total_tokens > 0
    assert client.calls == 1


@requires_key
def test_embedding_dimensions():
    vec = Embedder().embed("maximum LTV for a 2 unit investment property")
    assert len(vec) == get_settings().embed_dims
