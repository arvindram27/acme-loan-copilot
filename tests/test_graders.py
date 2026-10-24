"""Grader tests, offline: no key, no network, no DB.

- CodeGrader against TurnOutcome-like objects (constructed, no model calls).
- JevGrader against the fixture transport; escalation thresholds observed.
- The cascade only asks the judge for checks Jev itself was unsure about.
"""

import json

import pytest

from agentmem.eval.graders import (
    CHECKS, CodeGrader, Grade, JEV_QUESTIONS, JevGrader, LLMJudge,
    run_cascade,
)
from agentmem.providers.jev import JevClient, JevResult
from agentmem.settings import Settings


class Outcome:
    """Stand-in for TurnOutcome without importing the agent (kept light)."""

    def __init__(self, text, tool_results=None, tool_calls=None,
                 refs=None, facts=None):
        self.text = text
        self.tool_results = tool_results or []
        self.tool_calls = tool_calls or [
            {"name": t.get("name"), "source": t.get("source", "fresh")}
            for t in self.tool_results
        ]
        self.retrieved_refs = refs or []
        self.facts_used = facts or {}
        self.question = None


def test_code_required_tool_and_args():
    o = Outcome("$1,860", tool_results=[
        {"name": "affordability_calculator", "source": "fresh",
         "result": {"max_qualifying_payment": 1860.0, "dti_cap_pct": 45.0}},
    ])
    g = {g.check: g for g in CodeGrader().grade(
        {"required_tool": "affordability_calculator"}, o,
        question="i make 6800 and owe 1200")}
    assert g["required_tool_called"].pass_ is True
    assert g["tool_args_valid"].pass_ is True
    # 1860 is computed by the tool; question did not contain it
    assert g["number_provenance"].pass_ is True


def test_code_flags_poisoned_clamp():
    o = Outcome("you do not qualify", tool_results=[
        {"name": "affordability_calculator", "source": "fresh",
         "result": {"max_qualifying_payment": -1200.0, "dti_cap_pct": 45.0,
                    "constraint_used": "max_dti_pct=0 out of range"}},
    ])
    g = {g.check: g for g in CodeGrader().grade(
        {"required_tool": "affordability_calculator"}, o, question="x")}
    assert g["tool_args_valid"].pass_ is False


def test_code_must_include_exclude_and_cite():
    o = Outcome(
        "Your max payment is $1,860 [DTI Limits §1].",
        refs=[{"title": "DTI Limits", "chunk_index": 1, "preview": "..."}],
    )
    g = {g.check: g for g in CodeGrader().grade(
        {"must_include": ["1860"], "must_exclude": ["95000"], "cite": True},
        o, question="")}
    assert g["must_include"].pass_ is True
    assert g["must_exclude"].pass_ is True    # "95000" absent
    assert g["citation_present"].pass_ is True


def test_code_must_exclude_misses_stylized_variant():
    """95000 vs '95k': code is literal, the variant leaks through. This is
    exactly the check Jev's forbidden_absent noul exists for."""
    o = Outcome("You qualify at 95% LTV with your income of 95k.")
    g = {g.check: g for g in CodeGrader().grade(
        {"must_exclude": ["95%"]}, o, question="")}
    assert g["must_exclude"].pass_ is False   # literal "95%" caught
    g2 = {g.check: g for g in CodeGrader().grade(
        {"must_exclude": ["95000"]}, o, question="")}
    assert g2["must_exclude"].pass_ is True   # "95k" variant missed by code


def test_code_no_panic_on_pre_tool_answer():
    o = Outcome("About $3,060 — I computed it from your income.")
    g = CodeGrader().grade({"required_tool": "affordability_calculator"}, o,
                           question="income 6800 debts 1200")
    checks = {g.check: g for g in g}
    assert checks["required_tool_called"].pass_ is False
    assert checks["tool_args_valid"].pass_ is False  # no results at all
    assert checks["number_provenance"].pass_ is False  # nothing provenanced


FIXTURE = {
    "model": "jev-1.14.0",
    "answers": {
        "answer_grounded": {"type": "noul", "noul": 0.42},      # unsure!
        "forbidden_absent": {"type": "noul", "noul": 0.9},
        "failure_mode": {"type": "choice", "choice": "stale_fact",
                         "confidence": 0.62,
                         "probabilities": {"stale_fact": 0.62}},
        "answer_quality": {"type": "score", "score": 2.1, "confidence": 0.7,
                           "probabilities": {}},
    },
    "usage": {"input_tokens": 500, "output_tokens": 30},
}


class FakeTransport:
    def __init__(self, body):
        self.body = body
        self.last_payload = None

    def __call__(self, url, headers, payload):
        self.last_payload = payload
        return 200, self.body


def _jev_grader(body=FIXTURE, low=0.35, high=0.65):
    t = FakeTransport(body)
    s = Settings(jev_key="k", jev_escalate_low=low, jev_escalate_high=high,
                 _env_file=None)
    return JevGrader(JevClient(s, transport=t), low, high), t


def test_jev_batches_all_questions_in_one_call():
    jg, t = _jev_grader()
    o = Outcome("max payment 1860 [DTI §1]",
                tool_results=[{"name": "affordability_calculator",
                               "result": {"max_qualifying_payment": 1860.0}}],
                refs=[{"title": "DTI", "chunk_index": 1, "preview": "p"}])
    rows = jg.grade({"required_tool": "affordability_calculator"}, o,
                    question="q")
    assert len(rows) == 4
    assert set(JEV_QUESTIONS) <= set(CHECKS)
    qs = t.last_payload["questions"]
    assert set(qs) == set(JEV_QUESTIONS)
    state = t.last_payload["state"]
    assert state["tool_calls"][0]["result"]["max_qualifying_payment"] == 1860.0


def test_jev_escalation_band_marks_unsure_only():
    jg, _ = _jev_grader()
    o = Outcome("answer", tool_results=[], refs=[])
    rows = {r.check: r for r in jg.grade({}, o, question="q")}
    assert rows["answer_grounded"].escalated is True        # 0.42 in (0.35, 0.65)
    assert rows["failure_mode"].escalated is False          # conf 0.62
    assert rows["answer_quality"].escalated is False


def test_jev_low_confidence_escalates_choice():
    body = json.loads(json.dumps(FIXTURE))
    body["answers"]["failure_mode"]["confidence"] = 0.31
    jg, _ = _jev_grader(body)
    o = Outcome("a", tool_results=[], refs=[])
    rows = {r.check: r for r in jg.grade({}, o, question="q")}
    assert rows["failure_mode"].escalated is True


class FakeJudgeChat:
    def __init__(self):
        self.asked = None

    def complete(self, messages, **kw):
        self.asked = messages
        class R:
            text = json.dumps({
                "answer_grounded": {"pass": True},
                "answer_quality": {"score": 3},
            })
            total_tokens = 99
        return R()


def test_cascade_judge_targets_only_escalated_checks():
    jg, _ = _jev_grader()
    judge = FakeJudgeChat()
    o = Outcome("answer [DTI §1]", tool_results=[], refs=[
        {"title": "DTI", "chunk_index": 1, "preview": "p"}])
    rows = run_cascade(0, {}, o, "q", jev=jg, judge=LLMJudge(judge),
                       judge_checks={"answer_grounded"})
    got = {(r.grader, r.check) for r in rows}
    assert ("judge", "answer_grounded") in got       # jev was unsure (0.42)
    assert ("judge", "answer_quality") not in got    # jev was sure (conf .7)
    asked = judge.asked[0]["content"]
    assert "answer_grounded" in asked


def test_cascade_judge_skips_unvalidated_checks():
    """A check outside judge_checks never escalates, even when Jev is unsure."""
    jg, _ = _jev_grader()
    judge = FakeJudgeChat()
    o = Outcome("answer", tool_results=[], refs=[])
    rows = run_cascade(0, {}, o, "q", jev=jg, judge=LLMJudge(judge),
                       judge_checks=set())  # nothing validated
    got = {(r.grader, r.check) for r in rows}
    assert not any(g == "judge" for g, _ in got)
    assert judge.asked is None  # never invoked


def test_judge_prompt_embeds_check_definitions():
    """The judge must see the failure_mode taxonomy and the answer_quality
    rubric; the original prompt passed only check names (the bug that made
    the judge worse than Jev)."""
    jg, _ = _jev_grader()
    judge = FakeJudgeChat()
    o = Outcome("a", tool_results=[], refs=[])
    LLMJudge(judge).grade({}, o, "q", which=["failure_mode", "answer_quality"])
    content = judge.asked[0]["content"]
    assert "parametric_leak" in content       # a failure-mode name
    assert "0 = worst" in content             # rubric header
    assert "Wrong or harmful" in content      # rubric level 0 text


def test_grade_rows_unique_per_grader():
    jg, _ = _jev_grader()
    judge = LLMJudge(FakeJudgeChat())
    o = Outcome("a", tool_results=[], refs=[])
    rows = run_cascade(0, {"cite": True}, o, "", jev=jg, judge=judge)
    keys = [(r.grader, r.check) for r in rows]
    assert len(keys) == len(set(keys)), "UNIQUE(item, grader, check)"


@pytest.mark.parametrize("value,answer,include,exclude", [
    ("45", "The cap is 450000.", False, True),
    ("45", "The cap is 45 percent.", True, False),
    ("8750", "Income 87,500.", False, True),
    ("8750", "Income 8,750.", True, False),
    ("1200", "Debt -1,200.", False, True),
    ("-1200", "Debt -1,200.", True, False),
    ("3.5", "Rate 35%.", False, True),
    ("3.5", "Rate 3.5%.", True, False),
    ("2737", "Payment 2,737.50.", True, False),
])
def test_code_numeric_boundaries(value, answer, include, exclude):
    """must_include and must_exclude must not match by substring."""
    def grade(key):
        rows = CodeGrader().grade({key: [value]}, Outcome(answer), question="")
        return {r.check: r for r in rows}[key].pass_

    assert grade("must_include") is include
    assert grade("must_exclude") is exclude
