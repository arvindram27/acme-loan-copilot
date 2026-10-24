"""Fact audit, context snapshots, and run linkage (review #7, #8)."""

import random

import psycopg
import pytest

from agentmem.agent import MemoryAgent
from agentmem.db import migrate
from agentmem.eval.linkage import run_linkage
from agentmem.eval.workload import WORKLOAD_V1
from agentmem.settings import get_settings
from agentmem.units import entity

DIMS = 1536


class FakeEmbedder:
    def embed(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIMS
            v[hash(t) % DIMS] = 1.0
            out.append(v)
        return out


class FakeChat:
    def complete(self, messages, **kw):
        system = messages[0]["content"] if messages else ""

        class R:
            total_tokens = 3
            prompt_tokens = 2
            model = "fake"

        r = R()
        if "Extract factual statements" in system:
            r.text = '{"facts": []}'
        elif "tool-planning" in system or "decide the next step" in system:
            r.text = '{"action": "answer"}'
        else:
            r.text = "ok"
        return r


@pytest.fixture()
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    c.execute("DELETE FROM mem.fact_audit WHERE entity_id IN "
              "(SELECT entity_id FROM mem.entity WHERE user_id LIKE 'au-%')")
    for table in ("mem.turn", "mem.fact", "mem.entity", "mem.persona",
                  "mem.summary", "mem.cache"):
        c.execute(f"DELETE FROM {table} WHERE user_id LIKE 'au-%'")
    c.close()


def _user() -> str:
    return f"au-{random.randint(10**8, 10**9 - 1)}"


def test_fact_insert_and_supersede_are_audited(conn):
    u = _user()
    entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                      name=u, attr="income_monthly", value=1000)
    entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                      name=u, attr="income_monthly", value=2000)  # supersedes
    rows = conn.execute(
        "SELECT a.op, a.old_row, a.new_row FROM mem.fact_audit a "
        "JOIN mem.entity e ON e.entity_id = a.entity_id "
        "WHERE e.user_id = %s ORDER BY a.audit_id", (u,)).fetchall()
    ops = [r[0] for r in rows]
    assert "insert" in ops and "update" in ops
    upd = [r for r in rows if r[0] == "update"][0]
    assert upd[1] is not None and upd[2] is not None


def test_fact_delete_is_recorded(conn):
    u = _user()
    wrote = entity.write_fact(conn, user_id=u, thread_id="t", kind="applicant",
                              name=u, attr="income_monthly", value=1000)
    conn.execute("DELETE FROM mem.fact WHERE fact_id = %s", (wrote["fact_id"],))
    ops = conn.execute(
        "SELECT a.op FROM mem.fact_audit a "
        "JOIN mem.entity e ON e.entity_id = a.entity_id WHERE e.user_id = %s",
        (u,)).fetchall()
    assert ("delete",) in ops, "a deleted fact must leave an audit row"


def test_turn_stores_context_snapshot(conn):
    u = _user()
    agent = MemoryAgent(conn, user_id=u, chat=FakeChat(),
                        embedder=FakeEmbedder())
    out = agent.answer(thread_id="snap", text="hello")
    assert out.context_snapshot, "outcome must carry a snapshot"
    row = conn.execute(
        "SELECT context_snapshot FROM mem.turn "
        "WHERE user_id = %s AND thread_id = 'snap' AND role = 'assistant'",
        (u,)).fetchone()
    assert row is not None and row[0], "the assistant turn stores the snapshot"


def test_run_linkage_has_all_fields():
    link = run_linkage(get_settings(), WORKLOAD_V1)
    assert link["commit_hash"] and len(link["commit_hash"]) >= 7
    assert len(link["workload_hash"]) == 64
    assert link["prompt_version"]
    gc = link["grader_config"]
    assert "answer_grounded" in gc["jev_questions"]
    assert isinstance(gc["judge_validated"], list)
    assert "judge_model" in gc and "jev_model" in gc


def test_eval_run_accepts_linkage(conn):
    run_id = conn.execute(
        "INSERT INTO mem.eval_run "
        "(backend, model, workload, commit_hash, prompt_version, "
        " grader_config, workload_hash) "
        "VALUES ('memory', 'm', 'w', 'abc1234', 'v1', '{}'::jsonb, 'deadbeef') "
        "RETURNING run_id").fetchone()[0]
    row = conn.execute(
        "SELECT commit_hash, prompt_version, grader_config, workload_hash "
        "FROM mem.eval_run WHERE run_id = %s", (run_id,)).fetchone()
    assert row[0] == "abc1234" and row[1] == "v1"
    assert row[2] == {} and row[3] == "deadbeef"
    conn.execute("DELETE FROM mem.eval_run WHERE run_id = %s", (run_id,))


pytestmark = pytest.mark.db
