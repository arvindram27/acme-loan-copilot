"""Milestone 8 app tests: every page loads (200), served through TestClient."""

import pytest
from fastapi.testclient import TestClient

from agentmem.app.main import app
from agentmem.settings import get_settings

client = TestClient(app)


def test_index_redirects_to_chat():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 302)


def test_chat_page_loads():
    r = client.get("/chat")
    assert r.status_code == 200
    assert "Chat" in r.text


def test_memory_page_loads():
    r = client.get("/memory", params={"user_id": "maya"})
    assert r.status_code == 200
    assert "Memory inspector" in r.text


def test_data_page_loads():
    r = client.get("/data")
    assert r.status_code == 200
    assert "Data explorer" in r.text
    # selecting a real table renders columns/rows without error
    r2 = client.get("/data", params={"table": "turn", "limit": 5})
    assert r2.status_code == 200
    assert "Data explorer" in r2.text


def test_evals_page_loads():
    r = client.get("/evals")
    assert r.status_code == 200
    assert "Evaluation runs" in r.text


def test_eval_run_page_missing_is_404():
    r = client.get("/evals/999999999")
    assert r.status_code == 404


def test_chat_post_requires_key_or_errors_cleanly():
    """Without a key the POST must not 500 — it reports the error inline."""
    s = get_settings()
    if s.api_key:
        pytest.skip("MEM_API_KEY set — live chat path not exercised here")
    r = client.post("/chat", data={"user_id": "maya", "thread_id": "a",
                                   "message": "hello"})
    assert r.status_code == 200
    assert "disabled" in r.text
