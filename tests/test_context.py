"""Context-builder tests, offline: token estimation and budget trimming."""

from agentmem.units import context
from agentmem.units.context import assemble, estimate_tokens

SYSTEM = "You are the loan-officer copilot."


def _recalled(n):
    return [{"seq": i, "role": "user", "content": f"recalled turn {i} " + "x" * 40}
            for i in range(1, n + 1)]


def _chunks(n):
    return [{"chunk_id": i, "title": "Policy", "chunk_index": i,
             "text": f"chunk text {i} " + "y" * 40}
            for i in range(1, n + 1)]


def test_estimate_tokens_monotonic():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("x" * 100) > estimate_tokens("x" * 10)


def test_assemble_under_budget_keeps_everything():
    a = assemble(
        system_base=SYSTEM,
        persona_text="be concise",
        facts_text="income = 8750",
        summary={"covers_through_seq": 4, "text": "summed facts"},
        recalled=_recalled(2),
        chunks=_chunks(2),
        recent_before_current=[{"seq": 9, "role": "assistant", "content": "hi"}],
        max_context_tokens=100000,
    )
    assert a.summary_used is True
    assert a.recalled_turns == 2
    assert len(a.chunks) == 2
    assert a.context_tokens > 0
    assert "Policy excerpts" in a.system


def test_assemble_trims_lowest_priority_first():
    """With a tight budget the recall section is dropped before the KB, and
    the fixed sections (system, facts) are always kept."""
    a = assemble(
        system_base=SYSTEM,
        persona_text="",
        facts_text="income = 8750",
        summary=None,
        recalled=_recalled(3),
        chunks=_chunks(3),
        recent_before_current=[],
        max_context_tokens=estimate_tokens(SYSTEM) + estimate_tokens("## Known facts\nincome = 8750") + 5,
    )
    # recall and chunks could not fit the tiny remaining budget
    assert a.recalled_turns == 0
    assert a.chunks == []
    assert "## Known facts" in a.system
    assert "Policy excerpts" not in a.system


def test_assemble_no_summary_marks_false():
    a = assemble(
        system_base=SYSTEM,
        persona_text="", facts_text="", summary=None,
        recalled=[], chunks=[], recent_before_current=[],
        max_context_tokens=1000,
    )
    assert a.summary_used is False
    assert a.recalled_turns == 0
