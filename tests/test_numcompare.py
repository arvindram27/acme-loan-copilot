"""Numeric comparator regressions (review item 1).

Substring matching treated 45 as present inside 450000 and 8,750 inside 87,500;
these tests pin the value-aware behaviour.
"""

import pytest

from agentmem.eval.numcompare import contains_text, contains_value, numbers_in


@pytest.mark.parametrize("expected,answer,match", [
    ("45", "the cap is 450000", False),
    ("45", "the cap is 45 percent", True),
    ("8750", "income 87,500", False),
    ("8750", "income 8,750", True),
    ("1200", "debt -1,200", False),        # sign matters
    ("-1200", "debt -1,200", True),
    ("3.5", "rate 35%", False),
    ("3.5", "rate 3.5%", True),
    ("600", "limit 1600", False),          # magnitude matters
    ("2737", "payment 2,737.50", True),    # money headline tolerance
    ("0.125", "cost 0.125 points", True),
    ("0.125", "cost 0.12 points", False),  # rates stay tight
])
def test_contains_value_boundaries(expected, answer, match):
    assert contains_value(answer, expected) == match


def test_numbers_in_preserves_decimals_and_signs():
    vals = [str(n) for n in numbers_in("$-1,200 and 2,737.50 and 45%")]
    assert vals == ["-1200", "2737.50", "45"]


def test_contains_text_word_boundary():
    assert contains_text("the category is set", "cat") is False
    assert contains_text("the cat sat", "cat") is True
    assert contains_text("stale_fact", "stale") is False  # underscore is a word char
