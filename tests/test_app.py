"""App tests: every page loads (200), including the ladder, rungs, and the
backend switcher. Served through TestClient against the real database."""

import pytest
from fastapi.testclient import TestClient

from agentmem.app.main import app
from agentmem.settings import get_settings

client = TestClient(app)


def test_overview_page_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "Memory built twice" in r.text


def test_chat_page_loads():
    r = client.get("/chat")
    assert r.status_code == 200
    assert "Copilot" in r.text


def test_chat_page_langgraph_backend_loads():
    r = client.get("/chat", params={"backend": "langgraph"})
    assert r.status_code == 200
    assert "langgraph" in r.text


@pytest.mark.parametrize("n", list(range(8)))
def test_rung_pages_load(n):
    r = client.get(f"/rung/{n}")
    assert r.status_code == 200


def test_rung_8_redirects_to_measure():
    r = client.get("/rung/8", follow_redirects=False)
    assert r.status_code in (307, 302)


def test_unknown_rung_is_404():
    assert client.get("/rung/99").status_code == 404


def test_evals_page_loads():
    r = client.get("/evals")
    assert r.status_code == 200
    assert "Measure" in r.text


def test_eval_run_page_missing_is_404():
    assert client.get("/evals/999999999").status_code == 404


def test_data_page_loads():
    r = client.get("/data")
    assert r.status_code == 200
    assert "Data explorer" in r.text
    r2 = client.get("/data", params={"table": "turn", "limit": 5})
    assert r2.status_code == 200


def test_memory_page_loads():
    r = client.get("/memory", params={"user_id": "maya"})
    assert r.status_code == 200
    assert "Memory inspector" in r.text


def test_chat_post_requires_key_or_errors_cleanly():
    s = get_settings()
    if s.api_key:
        pytest.skip("MEM_API_KEY set — live chat path not exercised here")
    r = client.post("/chat", data={"user_id": "maya", "thread_id": "a",
                                   "message": "hello", "backend": "scratch"})
    assert r.status_code == 200
    assert "disabled" in r.text


pytestmark = pytest.mark.db
