"""Scorer versioning and the h3 duration rule (re-check 5, scoring item).

The frozen *question set* is v1 and is not touched. These tests pin the
separately versioned *scorer*: v1 stays lenient for the record, v2 rejects a
rate-lock answer that states 0.125 and cites a source but gets the period
missing or wrong.
"""

import pytest

from agentmem.eval.heldout_scorer import (
    SCORER_V1,
    SCORER_V2,
    Answer,
    has_seven_day_duration,
    load_frozen,
    score,
)

FROZEN = load_frozen()
QUESTIONS = {q["id"]: q for q in FROZEN.QUESTIONS}

CITE = ("\u00a7", ({"title": "Acme Product Sheet \u2014 Rate Lock Policy",
                    "chunk_index": 1},))


def _answer(text, *, cite=True, tools=()):
    if cite:
        return Answer(
            text=f"{text} [Acme Product Sheet \u2014 Rate Lock Policy \u00a71]",
            retrieved_refs=(CITE[1][0],),
            tool_calls=tuple(tools),
        )
    return Answer(text=text, retrieved_refs=(), tool_calls=tuple(tools))


H3 = QUESTIONS["h3-rag-lock"]


def test_v1_is_lenient_about_duration():
    """The recorded v1 rule only asked for the number and a citation."""
    answer = _answer("Extensions beyond grace cost 0.125 points.")
    verdict = score(H3, answer, version=SCORER_V1)
    assert verdict.passed is True
    assert "duration_ok" not in verdict.checks


@pytest.mark.parametrize("text", [
    "Extensions beyond the grace period cost 0.125 points.",
])
def test_v2_rejects_missing_duration(text):
    verdict = score(H3, _answer(text), version=SCORER_V2)
    assert verdict.passed is False
    assert verdict.checks["duration_ok"] is False
    assert any("duration" in f for f in verdict.failures)


@pytest.mark.parametrize("text", [
    "Extensions beyond the grace period cost 0.125 points per 30 days.",
    "Extensions beyond the grace period cost 0.125 points per two weeks.",
    "Extensions beyond the grace period cost 0.125 points per 14 days.",
    "Extensions beyond the grace period cost 0.125 points per month.",
])
def test_v2_rejects_wrong_duration(text):
    """Number and citation are both present; the period is wrong, so it fails."""
    verdict = score(H3, _answer(text), version=SCORER_V2)
    assert verdict.checks["must_include"]["0.125"] is True
    assert verdict.checks["cite_ok"] is True
    assert verdict.passed is False
    assert verdict.checks["duration_ok"] is False


@pytest.mark.parametrize("text", [
    "Extensions beyond the grace period cost 0.125 points per seven days.",
    "Extensions beyond the grace period cost 0.125 points per 7 days.",
    "Extensions beyond the grace period cost 0.125 points every 7-day period.",
    "Extensions beyond the grace period cost 0.125 points per week.",
    "Extensions beyond the grace period cost 0.125 points weekly.",
])
def test_v2_accepts_correct_duration_forms(text):
    verdict = score(H3, _answer(text), version=SCORER_V2)
    assert verdict.checks["duration_ok"] is True
    assert verdict.passed is True, verdict.failures


def test_v2_still_requires_the_citation():
    verdict = score(
        H3,
        _answer("Extensions cost 0.125 points per seven days.", cite=False),
        version=SCORER_V2,
    )
    assert verdict.passed is False
    assert verdict.checks["cite_ok"] is False
    assert "citation missing" in verdict.failures


def test_v2_still_requires_the_number():
    verdict = score(
        H3,
        _answer("Extensions beyond grace cost points per seven days."),
        version=SCORER_V2,
    )
    assert verdict.checks["must_include"]["0.125"] is False
    assert verdict.passed is False


def test_duration_helper_does_not_match_unrelated_periods():
    assert has_seven_day_duration("0.125 points per seven days") is True
    assert has_seven_day_duration("0.125 points per 7 days") is True
    assert has_seven_day_duration("0.125 points per 30 days") is False
    assert has_seven_day_duration("0.125 points per two weeks") is False
    assert has_seven_day_duration("0.125 points per month") is False


def test_v2_leaves_the_other_questions_unchanged():
    """Only h3 gains a check; the other four rules stay as frozen."""
    h1 = QUESTIONS["h1-tool"]
    h2 = QUESTIONS["h2-rag-ltv"]
    h4 = QUESTIONS["h4-supersede"]
    h5 = QUESTIONS["h5-cross-thread"]

    tool = ({"name": "affordability_calculator"},)
    assert score(h1, _answer("The most is $2,430/month.", tools=tool)).passed
    assert score(h1, _answer("The most is $2,430/month.")).passed is False
    assert score(h2, _answer("The maximum LTV is 85 percent \u00a71")).passed
    assert score(h4, _answer("Your income on file is $8,100.")).passed
    assert score(
        h4, _answer("Your income on file is $8,100, up from $7,400.")
    ).passed is False
    assert score(h5, _answer("Income $8,100; debts $900.")).passed
    # No override leaked into the other questions.
    for q in (h1, h2, h4, h5):
        assert "duration_ok" not in score(q, _answer("x")).checks
