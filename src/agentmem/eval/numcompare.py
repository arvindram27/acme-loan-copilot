"""Numeric-aware value comparison for graders and the demo check.

Substring matching is wrong for numbers: "45" occurs inside "450000" and
"8,750" inside "87,500", so a materially wrong answer can be graded correct.
This module parses numeric tokens out of the *raw* text (before any
normalisation) and compares values, so sign, magnitude and decimals matter.

Tolerance: an expected value matches the answer when the answer states a number
within half a unit of the expected value's last decimal place (0.5 * 10**-d).
That lets a money headline "2737" match "2,737.50" while keeping these all
different: 45 vs 450000, 8750 vs 87500, -1200 vs 1200, 3.5 vs 35.

Non-numeric expected values fall back to word-boundary matching, never
substring (so "600" does not match "1600", and "cat" does not match
"category").
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUM_RE = re.compile(r"(?<![\d])[-+]?\d[\d,]*(?:\.\d+)?")

__all__ = ["parse_number", "numbers_in", "contains_value", "contains_text"]


def parse_number(value) -> Decimal | None:
    """Coerce an expected value (int/float/str with $ , % ) to Decimal."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    text = (str(value).strip().replace(",", "").replace("$", "")
            .rstrip("%").strip())
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def numbers_in(text: str) -> list[Decimal]:
    """Every numeric token in the raw text, in order."""
    out: list[Decimal] = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        try:
            out.append(Decimal(raw))
        except InvalidOperation:
            continue
    return out


def _decimals(expected) -> int:
    text = str(expected).strip().replace(",", "").replace("$", "").rstrip("%")
    return len(text.split(".", 1)[1]) if "." in text else 0


def contains_value(text: str, expected, *, exact: bool = False) -> bool:
    """True when `text` states a number matching `expected`.

    Fuzzy (default): within half of the expected value's last decimal place, so
    a money headline "2737" matches "2,737.50". Exact: Decimal equality, used by
    `must_exclude` and the supersession guard where a false positive is harmful
    ("8" must not match "8.5").
    """
    exp = parse_number(expected)
    if exp is None:
        return contains_text(text, str(expected))
    if exact:
        return any(n == exp for n in numbers_in(text))
    tol = Decimal("0.5") * (Decimal(10) ** -_decimals(expected))
    return any(abs(n - exp) <= tol for n in numbers_in(text))


def contains_text(text: str, needle: str) -> bool:
    """Word-boundary containment for non-numeric expected values."""
    needle = (needle or "").strip()
    if not needle:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)",
                     text or "", re.IGNORECASE) is not None
