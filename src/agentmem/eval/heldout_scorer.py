"""Versioned scorer for the frozen held-out question set.

Question sets and scorers are versioned **independently**:

* a *question set* is identified by its frozen artifact (`heldout-frozen-v1`,
  commit `b92f0b1`): questions, expectations, seed message. It is never edited.
* a *scorer* is identified by `SCORER_VERSION` in this module. Revising the
  scoring rule does not create a new question set, and does not rewrite the
  result the old scorer produced.

Why this exists: v1 of the frozen script checks h3 (`rate-lock extension`) only
for the number `0.125` plus a citation, so an answer that states the right rate
but the wrong duration ("0.125 points per 30 days") or no duration at all still
passes. Scorer v2 adds the duration condition for that question and leaves every
other question's rule byte-for-byte equivalent to v1.

Two different things are produced from here, and they are labelled differently:

* **regrade** — stored answers re-scored, no model calls
  (`scripts/regrade_heldout.py`). Evidence that old answers survive the stricter
  rule; not a new run.
* **regression run** — the same five *known* questions executed once under the
  new scorer (`scripts/heldout_regression.py`). A regression check, **not** an
  unseen holdout: the questions are neither new nor hidden.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path

from agentmem.eval.numcompare import contains_value
from agentmem.settings import PROJECT_ROOT

SCORER_VERSION = "heldout-scorer-v2"
SCORER_V1 = "heldout-scorer-v1"
SCORER_V2 = SCORER_VERSION

FROZEN_SCRIPT = PROJECT_ROOT / "scripts" / "heldout_frozen.py"

#: Per-question tightening applied by v2, keyed by the frozen question id.
#: Kept here (not in the frozen script) so the question set stays untouched.
V2_OVERRIDES: dict[str, dict] = {
    "h3-rag-lock": {"require_duration_days": 7},
}

# "seven days", "7 days", "7-day", "per week", "one week", "a week", "weekly".
# A wrong duration ("30 days", "two weeks") deliberately does not match: the
# check is "a correct duration is stated", not "some duration is stated".
_SEVEN_DAYS = re.compile(
    r"\b(?:seven|7)\s*[-\s]\s*days?\b"
    r"|\b(?:per|each|every)\s+week\b"
    r"|\bone\s*[-\s]?week\b"
    r"|\ba\s+week\b"
    r"|\bweekly\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Answer:
    """The parts of a turn the scorer is allowed to see."""

    text: str
    retrieved_refs: tuple = ()
    tool_calls: tuple = ()

    @classmethod
    def from_mapping(cls, d: dict) -> "Answer":
        return cls(
            text=d.get("text", ""),
            retrieved_refs=tuple(d.get("retrieved_refs") or ()),
            tool_calls=tuple(d.get("tool_calls") or ()),
        )

    @classmethod
    def from_outcome(cls, out) -> "Answer":
        return cls(
            text=out.text,
            retrieved_refs=tuple(out.retrieved_refs or ()),
            tool_calls=tuple(out.tool_calls or ()),
        )


@dataclass(frozen=True)
class Verdict:
    passed: bool
    scorer_version: str
    checks: dict = field(default_factory=dict)
    failures: tuple = ()

    def to_dict(self) -> dict:
        return {
            "pass": self.passed,
            "scorer_version": self.scorer_version,
            "checks": self.checks,
            "failures": list(self.failures),
        }


def has_seven_day_duration(text: str) -> bool:
    """True when the text states a seven-day period in a recognised form."""
    return bool(_SEVEN_DAYS.search(text))


def _base_checks(question: dict, answer: Answer) -> tuple[dict, list[str]]:
    """The frozen v1 rule, unchanged."""
    include = {v: contains_value(answer.text, v)
               for v in question["must_include"]}
    exclude = {v: not contains_value(answer.text, v)
               for v in question["must_exclude"]}
    cite_ok = (
        ("\u00a7" in answer.text and bool(answer.retrieved_refs))
        if question["cite"] else True
    )
    names = [c.get("name") for c in answer.tool_calls]
    required = question.get("require_tool")
    tool_ok = (required in names) if required else True

    checks = {
        "must_include": include,
        "must_exclude_absent": exclude,
        "cite_ok": cite_ok,
        "tool_ok": tool_ok,
        "tool_calls": names,
    }
    failures = []
    failures += [f"missing {v}" for v, ok in include.items() if not ok]
    failures += [f"stale {v} present" for v, ok in exclude.items() if not ok]
    if not cite_ok:
        failures.append("citation missing")
    if not tool_ok:
        failures.append(f"required tool {required} not called")
    return checks, failures


def score_v1(question: dict, answer: Answer) -> Verdict:
    """Reproduce the frozen script's original rule exactly."""
    checks, failures = _base_checks(question, answer)
    return Verdict(
        passed=not failures,
        scorer_version=SCORER_V1,
        checks=checks,
        failures=tuple(failures),
    )


def score(
    question: dict,
    answer: Answer,
    *,
    version: str = SCORER_VERSION,
) -> Verdict:
    """Score one answer under the requested scorer version."""
    if version == SCORER_V1:
        return score_v1(question, answer)
    if version != SCORER_V2:
        raise ValueError(f"unknown scorer version: {version}")

    checks, failures = _base_checks(question, answer)
    override = V2_OVERRIDES.get(question["id"], {})

    days = override.get("require_duration_days")
    if days is not None:
        # Only seven-day phrasing is recognised, because that is the policy term
        # these questions probe; a numeric "30 days" therefore fails.
        duration_ok = has_seven_day_duration(answer.text) if days == 7 else False
        checks["duration_ok"] = duration_ok
        if not duration_ok:
            failures.append(f"duration missing or wrong (expected {days} days)")

    return Verdict(
        passed=not failures,
        scorer_version=SCORER_V2,
        checks=checks,
        failures=tuple(failures),
    )


def load_frozen(path: Path = FROZEN_SCRIPT):
    """Load the frozen question set without modifying it.

    Reads `QUESTIONS` and `SEED_MESSAGE` from the committed v1 script, so the
    regression run provably uses the same questions that were frozen at
    `b92f0b1` rather than a copy that could drift.
    """
    spec = importlib.util.spec_from_file_location("heldout_frozen_v1", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load frozen set from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
