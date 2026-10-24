"""Graders: code, Jev (decision model), and the LLM judge cascade.

The rule of authority: code decides what code can prove deterministically
(tool called, args valid, numbers provenanced, forbidden values absent,
citations present). Jev grades the judgment checks against the trace as a
structured state — one batched call per item. Any Jev answer that lands in
the uncertainty band (or carries low confidence) escalates to the LLM judge.

Every grader returns Grade rows with the same check names, so agreement
(kappa against validated labels) and escalation rates are reportable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agentmem.eval import numcompare
from agentmem.units import toolbox

CHECKS = [
    # code-gradable (deterministic)
    "required_tool_called",
    "tool_args_valid",
    "number_provenance",
    "must_include",
    "must_exclude",
    "citation_present",
    # judgment checks (jev, escalate to judge on uncertainty)
    "answer_grounded",
    "forbidden_absent",
    "failure_mode",
    "answer_quality",
]

_FAILURE_MODES = {
    "none": "No defect found.",
    "parametric_leak": (
        "Asserts a policy number (rate, limit, percentage) that does not "
        "appear in `retrieved` or `tool_calls[*].result` — knowledge from "
        "the model's training, not from context."
    ),
    "poisoned_tool_input": (
        "A tool call carried an out-of-range or nonsense argument and the "
        "answer presents the tool's degenerate output as fact."
    ),
    "stale_fact": (
        "Presents a superseded or forbidden value from `must_be_declined` "
        "as current."
    ),
    "ungrounded_arithmetic": (
        "States a number that cannot be traced to `question`, "
        "`known_facts`, or a tool result."
    ),
    "omits_tool_result": (
        "The tool ran and produced an authoritative number, but the answer "
        "never states it (evades, hedges, or changes the subject)."
    ),
    "memory_miss": (
        "The user's own stored fact exists in `known_facts` or the supplied "
        "trace, but the answer claims not to have it."
    ),
    "scope_leak": (
        "Uses information that belongs to another user or thread."
    ),
}

_CITE_RE = re.compile(r"\[[^\]§]+§\d+\]")


def _norm(text: str) -> str:
    return (text or "").replace("$", "").replace(",", "").replace("%", "")


@dataclass
class Outcome:
    """The graded trace for one answer — the duck-typed shape of TurnOutcome
    (or a synthetic construction) that graders read. Kept here so labels
    and tests never import the agent to build states."""

    text: str = ""
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    retrieved_refs: list = field(default_factory=list)
    facts_used: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.tool_results and not self.tool_calls:
            self.tool_calls = [
                {"name": t.get("name"), "source": t.get("source", "fresh")}
                for t in self.tool_results
            ]


@dataclass
class Grade:
    check: str
    grader: str                       # code | jev | judge
    pass_: bool | None                # None for score/type grades
    value: object = None              # score float or failure-mode choice
    probability: float | None = None  # jev noul
    confidence: float | None = None
    escalated: bool = False
    rationale: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    tokens: int = 0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CodeGrader — deterministic, free, authoritative for what it can express
# ---------------------------------------------------------------------------

def _numbers_in_results(tool_results: list[dict]) -> list[float]:
    nums: list[float] = []
    for t in tool_results or []:
        res = t.get("result") or {}
        for v in res.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nums.append(float(v))
    return nums


class CodeGrader:
    """Grades (expected-labels, outcome) pairs mechanically. Expected is the
    item's `expected` jsonb: required_tool, must_include, must_exclude, cite.

    label_source notes where the label set came from ('constructed' or
    'hand'), carried into rows for the kappa table."""

    label_source = "code"

    def grade(self, expected: dict, outcome, question: str = "") -> list[Grade]:
        out: list[Grade] = []
        raw = outcome.text if outcome else ""
        answer = _norm(raw)
        question = _norm(question)
        required = expected.get("required_tool")

        if required:
            calls = outcome.tool_calls if outcome else []
            names = [t.get("name") for t in (calls or [])]
            out.append(Grade(
                check="required_tool_called", grader="code",
                pass_=required in names,
                rationale=(f"required {required}; ran {names}"),
            ))
            # args validity: no errors, and no degenerate output. The clamp
            # NOTE (constraint_used) is the tool self-correcting a bad input —
            # that is desired, not a defect. A degenerate result (negative
            # qualifying payment) is what must be flagged.
            results = outcome.tool_results or [] if outcome else []
            errored = any(t.get("source") == "error" for t in results)
            degenerate = any(
                t.get("name") == required
                and isinstance((t.get("result") or {}).get(
                    "max_qualifying_payment"), (int, float))
                and (t["result"]["max_qualifying_payment"] < 0)
                for t in results
            )
            out.append(Grade(
                check="tool_args_valid", grader="code",
                pass_=bool(results) and not errored and not degenerate,
                rationale=("degenerate result" if degenerate
                           else ("tool errored" if errored else "clean")),
            ))

        if "must_include" in expected:
            # Numeric-aware: 45 must not match 450000, 8750 not 87500.
            miss = [v for v in expected["must_include"]
                    if not numcompare.contains_value(raw, v)]
            out.append(Grade(
                check="must_include", grader="code", pass_=not miss,
                rationale=("absent from answer" if miss else "present"),
            ))

        if "must_exclude" in expected:
            leak = [v for v in expected["must_exclude"]
                    if numcompare.contains_value(raw, v)]
            out.append(Grade(
                check="must_exclude", grader="code", pass_=not leak,
                rationale=(f"leaked {leak}" if leak else "absent"),
            ))

        if required:
            # number_provenance: at least one tool-computed number reaches
            # the answer. Requiring EVERY internal field would be noise
            # (dti_cap_pct is bookkeeping, not an answer obligation); a
            # fabricated number is failure_mode's job, a poisoned result is
            # tool_args_valid's, a wrong value is must_include's.
            nums = _numbers_in_results(outcome.tool_results or [])
            reached = [n for n in nums if numcompare.contains_value(raw, n)]
            out.append(Grade(
                check="number_provenance", grader="code",
                pass_=bool(nums) and bool(reached),
                rationale=("no tool number in answer" if not reached
                           else f"reached {reached}"),
            ))

        if expected.get("cite"):
            refs = outcome.retrieved_refs or [] if outcome else []
            cited = bool(_CITE_RE.search(outcome.text or "")) or any(
                r["title"].lower() in answer for r in refs
            )
            out.append(Grade(
                check="citation_present", grader="code", pass_=cited,
                rationale=("cited" if cited
                           else f"{len(refs)} refs available, none cited"),
            ))
        return out


# ---------------------------------------------------------------------------
# JevGrader — one batched System One call per item; nouls escalate on doubt
# ---------------------------------------------------------------------------

# Single-source answer-quality rubric. labels.py assigns per-mutation levels
# against this table, and BOTH Jev and the LLM judge receive exactly these
# strings, so the two graders and the labels share one definition.
ANSWER_QUALITY_RUBRIC = [
    "Wrong or harmful: misstates a central fact or policy, presents a "
    "degenerate result as valid, or leaks another user's data as fact.",
    "Wrong, evasive, or incomplete on the core question: the central "
    "number or fact is stale, invented, missing, or dodged.",
    "Core answer correct, but with a grounding gap: an extra unsupported "
    "claim, or a missing citation.",
    "Correct, grounded in the trace, and reads well.",
]

JEV_QUESTIONS = {
    "answer_grounded": {
        "type": "noul",
        "instructions": (
            "Every factual claim in `answer` (rates, limits, percentages, "
            "dollar amounts, ratios) is supported by a policy excerpt in "
            "`retrieved`, by a value in `tool_calls[*].result`, by "
            "`known_facts`, or by `question`. A claim that appears in none "
            "of these is ungrounded. Statements with no such factual claim "
            "count as true."
        ),
    },
    "forbidden_absent": {
        "type": "noul",
        "instructions": (
            "None of the values in `must_be_declined` is presented in "
            "`answer` as a current fact or figure. If `must_be_declined` "
            "is empty, answer YES."
        ),
    },
    "failure_mode": {
        "type": "choice",
        "instructions": (
            "Which single defect does `answer` have, judged against the "
            "trace? Prefer the most specific defect; if several, the one "
            "that caused the others."
        ),
        "criteria": _FAILURE_MODES,
    },
    "answer_quality": {
        "type": "score",
        "instructions": (
            "How good is `answer` as a loan-officer copilot reply on the "
            "trace, judged only by what the trace shows?"
        ),
        "criteria": ANSWER_QUALITY_RUBRIC,
    },
}


class JevGrader:
    """State starts from the trace, not from trust: tool args and results,
    retrieved previews, and the item's forbidden values are all in the
    JSON state, so questions can point at paths backticked by name."""

    def __init__(self, jev, escalate_low: float, escalate_high: float):
        self.jev = jev
        self.low = escalate_low
        self.high = escalate_high

    @staticmethod
    def state(expected: dict, outcome, question: str) -> dict:
        return {
            "question": question,
            "answer": outcome.text,
            "required_tool": expected.get("required_tool"),
            "tool_calls": [
                {"name": t["name"], "args": t.get("args"),
                 "source": t.get("source"), "result": t.get("result")}
                for t in (outcome.tool_results or [])
            ],
            "retrieved": [
                {"ref": f"{r['title']} §{r['chunk_index']}",
                 "excerpt": r["preview"]}
                for r in (outcome.retrieved_refs or [])
            ],
            "known_facts": outcome.facts_used or {},
            "must_be_declined": expected.get("must_exclude", []),
        }

    def grade(self, expected: dict, outcome, question: str = "") -> list[Grade]:
        result = self.jev.evaluate(
            self.state(expected, outcome, question), JEV_QUESTIONS
        )
        rows: list[Grade] = []
        for qid, a in result.answers.items():
            common = dict(
                grader="jev", latency_ms=result.latency_ms,
                meta={"cost_usd": result.cost_usd / max(len(result.answers), 1)},
            )
            if a.type == "noul":
                escalated = self.low < a.value < self.high
                rows.append(Grade(
                    check=qid, pass_=a.value >= 0.5,
                    probability=a.value, escalated=escalated,
                    tokens=result.input_tokens + result.output_tokens,
                    **common,
                ))
            elif a.type == "choice":
                escalated = a.confidence is not None and a.confidence < 0.5
                rows.append(Grade(
                    check=qid, pass_=None, value=a.value,
                    confidence=a.confidence, escalated=escalated,
                    **common,
                ))
            else:  # score
                escalated = a.confidence is not None and a.confidence < 0.5
                rows.append(Grade(
                    check=qid, pass_=None, value=a.value,
                    confidence=a.confidence, escalated=escalated,
                    tokens=result.input_tokens + result.output_tokens,
                    **common,
                ))
        return rows


# ---------------------------------------------------------------------------
# LLMJudge — escalation only: decides exactly the checks it is handed
# ---------------------------------------------------------------------------

def _check_spec(check: str) -> str:
    """The definition a grader needs to decide one check. The judge prompt
    must carry this: without it the judge was asked to classify into a
    taxonomy it could not see (the original 13%-accurate failure_mode)."""
    if check == "failure_mode":
        lines = ["Failure modes (pick exactly one):"]
        lines += [f"- {k}: {v}" for k, v in _FAILURE_MODES.items()]
        return "\n".join(lines)
    if check == "answer_quality":
        lines = ["Answer quality rubric (0 = worst .. 3 = best):"]
        lines += [f"{i}: {t}" for i, t in enumerate(ANSWER_QUALITY_RUBRIC)]
        return "\n".join(lines)
    q = JEV_QUESTIONS.get(check)
    if q and q.get("instructions"):
        return f"Definition: {q['instructions']}"
    return ""


JUDGE_PROMPT = (
    "You are the tie-breaking grader for a loan copilot. Given the trace "
    "(question, answer, tool calls and results, retrieved policy excerpts, "
    "known facts, forbidden values), decide ONLY the named checks.\n\n"
    "{specs}\n\n"
    "Respond ONLY with JSON: "
    '{{"<check_name>": {{"pass": true|false}}}} for yes/no checks, '
    '{{"failure_mode": {{"choice": "<one failure mode>"}}}}, and '
    '{{"answer_quality": {{"score": <int 0-3>}}}}. One entry per named check.'
)


class LLMJudge:
    def __init__(self, chat, model: str | None = None):
        self.chat = chat
        self.model = model  # None -> the chat client's default (same family)

    def grade(self, expected: dict, outcome, question: str = "",
              which: list[str] | None = None) -> list[Grade]:
        state = JevGrader.state(expected, outcome, question)
        checks = which or ["answer_grounded", "forbidden_absent",
                           "failure_mode", "answer_quality"]
        specs = "\n\n".join(f"--- {c} ---\n{_check_spec(c)}" for c in checks)
        messages = [
            {"role": "system",
             "content": JUDGE_PROMPT.format(specs=specs)},
            {"role": "user", "content": json.dumps(state)},
        ]
        out = self.chat.complete(
            messages, temperature=0.0, max_tokens=400,
            response_format={"type": "json_object"},
            model=self.model,
        )
        rows: list[Grade] = []
        try:
            verdicts = json.loads(out.text)
        except json.JSONDecodeError:
            return []
        for check in checks:
            v = verdicts.get(check)
            if v is None:
                continue
            if check == "failure_mode":
                rows.append(Grade(check=check, grader="judge", pass_=None,
                                  value=v.get("choice"), escalated=True))
            elif check == "answer_quality":
                rows.append(Grade(check=check, grader="judge", pass_=None,
                                  value=float(v.get("score", 0)),
                                  escalated=True,
                                  tokens=out.total_tokens))
            else:
                rows.append(Grade(check=check, grader="judge",
                                  pass_=bool(v.get("pass")), escalated=True,
                                  tokens=out.total_tokens))
        return rows


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

# Checks the judge is authorized to overrule Jev on. Populated from the
# validation run: a check is added only when the judge's agreement reaches
# 0.6 with the fixed prompt (kappa for noul checks, exact/ordinal for choice
# and score). Empty means the judge is never trusted; Jev's low-confidence
# answer is kept and flagged escalated=True (which now means "low confidence",
# not "the judge ran"). Run scripts/validate_graders.py and fill this in from
# its output rather than hand-editing it.
JUDGE_VALIDATED: set[str] = set()


def run_cascade(item_no: int, expected: dict, outcome, question: str, *,
                jev: JevGrader | None, judge: LLMJudge | None,
                judge_checks: set[str] | None = None) -> list[Grade]:
    """code always; jev on judgment checks; judge only for checks it is
    validated on (judge_checks, or the module JUDGE_VALIDATED set)."""
    rows = CodeGrader().grade(expected, outcome, question=question)
    if jev is not None and jev.jev.has_key:
        rows += jev.grade(expected, outcome, question)
        unsure = [g.check for g in rows if g.grader == "jev" and g.escalated]
        allowed = JUDGE_VALIDATED if judge_checks is None else judge_checks
        to_judge = [c for c in unsure if c in allowed]
        if judge is not None and to_judge:
            rows += judge.grade(expected, outcome, question, which=to_judge)
    return rows
