"""Unit 5 contract tests: toolbox, cache, offload — no model calls needed."""

import json
from pathlib import Path

import pytest
import psycopg

from agentmem.db import migrate
from agentmem.settings import get_settings, PROJECT_ROOT
from agentmem.units import toolbox


@pytest.fixture(scope="module")
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    c.execute(
        "DELETE FROM mem.tool_call WHERE user_id LIKE 'm4-%'"
    )
    c.execute("DELETE FROM mem.tool WHERE embedding IS NULL AND name IN "
              "('affordability_calculator','rate_lookup','ltv_check','policy_full_dump')")
    c.execute("DELETE FROM mem.persona WHERE user_id LIKE 'm4-%'")
    c.close()


def _user():
    import random
    return f"m4-{random.randint(10**8, 10**9 - 1)}"


def test_args_hash_is_order_insensitive():
    a = toolbox.args_hash("rate_lookup", {"term_years": 30, "lock_days": 60})
    b = toolbox.args_hash("rate_lookup", {"lock_days": 60, "term_years": 30})
    assert a == b


def test_ttl_zero_tools_never_cache(conn):
    u = _user()
    first = toolbox.run_tool(conn, user_id=u, thread_id="t1",
                             name="affordability_calculator",
                             args={"income_monthly": 6800, "monthly_debts": 1200})
    second = toolbox.run_tool(conn, user_id=u, thread_id="t1",
                              name="affordability_calculator",
                              args={"income_monthly": 6800, "monthly_debts": 1200})
    assert first["source"] == "fresh"
    assert second["source"] == "fresh", "ttl=0 tools must always recompute"
    assert first["result_for_model"] == second["result_for_model"]


def rate_args():
    return {"term_years": 30, "lock_days": 60}


def test_rate_lookup_caches_within_ttl(conn):
    u = _user()
    first = toolbox.run_tool(conn, user_id=u, thread_id="t2",
                             name="rate_lookup", args=dict(rate_args()))
    second = toolbox.run_tool(conn, user_id=u, thread_id="t2",
                              name="rate_lookup", args=dict(rate_args()))
    assert [first["source"], second["source"]] == ["fresh", "cache"]
    locked = json.loads(second["result_for_model"])
    assert locked["rate_pct"] == 6.25 and locked["lock_days"] == 60


def test_expired_result_recomputes(conn):
    u = _user()
    # run once, backdate, expect fresh again
    first = toolbox.run_tool(conn, user_id=u, thread_id="t3",
                             name="rate_lookup", args=dict(rate_args()))
    conn.execute(
        """
        UPDATE mem.tool_call
        SET computed_at = now() - interval '2 hours'
        WHERE call_id = %s
        """,
        (first["call_id"],),
    )
    second = toolbox.run_tool(conn, user_id=u, thread_id="t3",
                              name="rate_lookup", args=dict(rate_args()))
    assert second["source"] == "fresh"


def test_different_args_miss(conn):
    u = _user()
    toolbox.run_tool(conn, user_id=u, thread_id="t4", name="rate_lookup",
                     args={"term_years": 30, "lock_days": 30})
    other = toolbox.run_tool(conn, user_id=u, thread_id="t4",
                             name="rate_lookup",
                             args={"term_years": 30, "lock_days": 90})
    assert other["source"] == "fresh"


def test_offload(conn):
    """The corpus dump must be offloaded to an artifact, not stored inline."""
    u = _user()
    out = toolbox.run_tool(conn, user_id=u, thread_id="t5",
                           name="policy_full_dump", args={})
    assert out["offloaded"], "37 chunks should exceed the 8KB threshold"
    assert out["result_bytes"] > toolbox.OFFLOAD_BYTES
    row = conn.execute(
        "SELECT artifact_path FROM mem.tool_call WHERE call_id = %s",
        (out["call_id"],),
    ).fetchone()
    path = toolbox.ARTIFACT_DIR / row[0]
    assert path.exists()
    full = json.loads(path.read_text("utf-8"))
    assert full["chunks"] >= 30
    # cached re-run reloads from the artifact
    _ = toolbox.run_tool(conn, user_id=u, thread_id="t5", name="policy_full_dump", args={})
    cached_call = conn.execute(
        "SELECT call_id FROM mem.tool_call WHERE user_id=%s AND artifact_path=%s",
        (u, row[0]),
    ).fetchall()
    assert cached_call


def test_tool_error_is_logged(conn):
    u = _user()
    with pytest.raises(ValueError):
        toolbox.run_tool(conn, user_id=u, thread_id="t6", name="rate_lookup",
                         args={"term_years": 7})
    logged = conn.execute(
        "SELECT source FROM mem.tool_call WHERE user_id = %s", (u,)
    ).fetchall()
    assert ("error",) in logged


def test_search_tools_ranks_live_registry(conn, monkeypatch):
    """Offline embedding inserted by hand; ranking is database-side."""
    conn.execute(
        "UPDATE mem.tool SET embedding = %s::text::vector WHERE name = 'rate_lookup'",
        (toolbox._vec([0.0] * 1535 + [1.0]),),
    )
    hits = toolbox.search_tools(conn, [0.0] * 1535 + [1.0], k=1)
    assert hits and hits[0]["name"] == "rate_lookup"
