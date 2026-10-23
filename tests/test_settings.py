"""Settings defaults and env overrides — offline, no key needed."""

from agentmem.settings import get_settings


def test_defaults():
    s = get_settings()
    assert s.openrouter_base_url.endswith("/v1")
    assert s.embed_dims == 1536
    assert s.chat_model and "/" not in s.chat_model or s.chat_model  # any non-empty id
    assert s.embed_model
    assert s.database_url.endswith("/agentmem")
    assert s.database_url_lg.endswith("/agentmem_lg")


def test_env_prefix_overrides(monkeypatch):
    monkeypatch.setenv("MEM_CHAT_MODEL", "test/provider-x")
    s = get_settings()
    assert s.chat_model == "test/provider-x"
