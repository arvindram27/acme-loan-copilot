"""Unit 4 + extraction-validation contract tests, no model calls.

Knowledge searches use injected one-hot embeddings; extraction validation is
pure and therefore offline by design.
"""

import pytest
import psycopg

from agentmem.db import migrate
from agentmem.settings import get_settings
from agentmem.units import extract, knowledge


@pytest.fixture(scope="module")
def conn():
    migrate(get_settings().database_url)
    c = psycopg.connect(get_settings().database_url, autocommit=True)
    yield c
    c.execute("DELETE FROM mem.doc WHERE title LIKE 'TD-%'")
    c.close()


def _fake(slot: int, dims: int = 1536) -> list[float]:
    v = [0.0] * dims
    v[slot] = 1.0
    return v


def _add_document_like(conn, title, text, slot):
    return knowledge.add_doc(conn, title=title, version="0", text=text,
                             embedding_fn=lambda s: _fake(slot))


# --- knowledge base ----------------------------------------------------------

def test_add_doc_counts_and_is_idempotent(conn):
    a = _add_document_like(conn, "TD-1", "First rule about LTV.\n\nSecond paragraph.", 0)
    b = _add_document_like(conn, "TD-1", "First rule about LTV.\n\nSecond paragraph.", 0)
    assert a["chunks"] == b["chunks"] == 2
    n = conn.execute(
        "SELECT COUNT(*) FROM mem.chunk WHERE doc_id IN "
        "(SELECT doc_id FROM mem.doc WHERE title='TD-1')"
    ).fetchone()[0]
    assert n == 2, "re-seed replaces chunks rather than duplicating"


def test_keyword_search_ranks_and_scope(conn):
    _add_document_like(conn, "TD-2", "Gift funds allowed for a family member donor.", 3)
    hits = knowledge.keyword_search(conn, "gift funds donor", k=3)
    assert hits, "keyword search should match TD-2"
    assert hits[0]["title"] == "TD-2"


def test_vector_search_min_score(conn):
    _add_document_like(conn, "TD-3", "unrelated filler text about nothing", 7)
    close = knowledge.vector_search(conn, _fake(7), k=5, min_score=0.9)
    far = knowledge.vector_search(conn, _fake(0), k=5, min_score=0.9)
    assert [h["title"] for h in close if h["title"] == "TD-3"]
    assert not [h for h in far if h["title"] == "TD-3"], \
        "min_score must exclude unrelated chunks"


def test_hybrid_fusion(conn):
    """A chunk relevant to both keyword and vector beats either side alone."""
    _add_document_like(conn, "TD-4",
                       "Maximum LTV for an investment two-unit property.", 11)
    hits = knowledge.hybrid_search(conn, query="investment two-unit LTV",
                                   query_embedding=_fake(11), k=3)
    assert hits[0]["title"] == "TD-4" and hits[0]["by"] == "hybrid"


def test_oversized_paragraph_splits(conn):
    text = (" ".join(["sentence per word filler"] * 200))  # > CHUNK_TARGET
    out = knowledge.add_doc(conn, title="TD-5", version="0", text=text,
                            embedding_fn=lambda s: _fake(0))
    assert out["chunks"] > 1


# --- extraction validation (pure) -------------------------------------------

def test_possessive_name_is_normalized():
    """'Ravi Chandra's annual income' must not create entity 'Ravi Chandras'."""
    valid, _ = extract.validate_facts([
        {"kind": "applicant", "name": "Ravi Chandra's",
         "attr": "income_annual", "value": 84000},
    ])
    assert valid[0]["name"] == "Ravi Chandra"


def test_valid_numeric_extraction():
    valid, dropped = extract.validate_facts([
        {"kind": "applicant", "name": "Jane Doe", "attr": "income_annual",
         "value": "105,000"},
        {"kind": "property", "name": "48 Elm St", "attr": "units",
         "value": 2},
    ])
    assert dropped == []
    assert valid[0]["value"] == 105000 and valid[1]["value"] == 2


def test_allowlist_and_coercion_drops():
    valid, dropped = extract.validate_facts([
        {"kind": "applicant", "name": "X", "attr": "favorite_color", "value": "teal"},
        {"kind": "kitchen", "name": "X", "attr": "income_monthly", "value": 1},
        {"kind": "applicant", "name": "", "attr": "income_monthly", "value": 1},
        {"kind": "applicant", "name": "X", "attr": "income_monthly", "value": "high"},
        {"kind": "applicant", "name": "X", "attr": "first_time_buyer", "value": "yes"},
    ])
    assert len(dropped) == 4 and len(valid) == 1
    assert valid[0]["attr"] == "first_time_buyer" and valid[0]["value"] is True
