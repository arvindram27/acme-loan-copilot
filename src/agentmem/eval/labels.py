"""Constructed labels by defect injection (Part 7 part 2, adapted).

Labeled workload turns × mutations = ~120 items with KNOWN grades. Each
mutation flips exactly the checks it targets and leaves the rest true, which
is what makes k-computable agreement meaningful:

  clean            everything passes; failure_mode=none; quality 3
  parametric_leak  appends an invented market number -> answer_grounded False
  invent_number    appends an invented personal figure -> answer_grounded F
  stale_leak       re-presents a forbidden value -> must_exclude +
                   forbidden_absent False (only for turns that grade it)
  strip_citations  removes [.. §n] (only for cite=True turns)
  memory_miss      claims nothing on file (entity turns only) -> must_include F
  drop_tool_number erases the authoritative number (tool turns)
  poisoned_tool    out-of-range arg -> tool emits garbage, answer repeats it

Mutations apply only where they make sense; total lands near 120 and the
count is reported, not fudged.
"""

from __future__ import annotations

import re

from agentmem.eval.graders import Outcome
from agentmem.eval.workload import TOOL_FACTS, WORKLOAD_V1, label_of
SELECTION = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 20,
             21, 23, 25, 28, 29]

_KB = {
    6:  ([{"title": "Debt-to-Income", "chunk_index": 0,
           "preview": "Acme's maximum debt-to-income ratio for conventional "
                      "loans is 45 percent. The limit rises to 50 percent "
                      "when a 760 FICO, six months reserves and a one-unit "
                      "owner-occupied property are all present."}],
         "Acme's maximum DTI ratio for a conventional loan is 45 percent "
         "[Acme Underwriting Guide - Debt-to-Income §0]."),
    7:  ([{"title": "LTV Limits", "chunk_index": 0,
           "preview": "For investment properties the limits are lower: the "
                      "maximum LTV is 80 percent on a one-unit investment "
                      "property. For a two-unit or larger investment "
                      "property, the maximum LTV is 75 percent."}],
         "For a 2-unit investment property, Acme's maximum LTV is 75 "
         "percent [Acme Underwriting Guide - LTV Limits §0]."),
    8:  ([{"title": "Rate Lock Policy", "chunk_index": 0,
           "preview": "A lock may be extended once for free within a "
                      "five-day grace period after expiration. Extensions "
                      "beyond the grace period cost 0.125 points per seven "
                      "days."}],
         "Yes - one free extension inside the five-day grace period after "
         "expiration; beyond it an extension costs 0.125 points per seven "
         "days [Acme Product Sheet - Rate Lock Policy §0]."),
    16: ([{"title": "Credit Score Minimums", "chunk_index": 0,
           "preview": "The minimum representative FICO for conventional "
                      "conforming loans at Acme is 620. Borrowers below 620 "
                      "cannot be manually underwritten."}],
         "Acme's minimum representative FICO for a conventional conforming "
         "loan is 620 [Acme Underwriting Guide - Credit Score Minimums §0]."),
    29: ([{"title": "Rate Lock Policy", "chunk_index": 0,
           "preview": "Acme offers thirty-, forty-five- and sixty-day rate "
                      "locks at par on conforming products. Ninety-day "
                      "locks carry a 0.25 point fee collected at lock."}],
         "Acme offers 30-, 45- and 60-day locks at par; only the 90-day "
         "lock costs points, at 0.25 [Acme Product Sheet - Rate Lock "
         "Policy §0]."),
}

_ENTITY = {
    3:  ({"maya.income_monthly": 8750},
         "Your monthly income on file is 8,750."),
    13: ({"maya.income_monthly": 8750, "maya.monthly_debts": 600},
         "On file for you: monthly income 8,750, and 600 in monthly debt "
         "payments after the car loan cleared."),
    14: ({"maya.income_monthly": 8750, "maya.monthly_debts": 600},
         "With your current file (income 8,750, debts 600), Acme's DTI cap "
         "supports at most 3,337.50 per month."),
    15: ({"maya.monthly_debts": 600},
         "Your debts on file are 600 - the car loan cleared, you told me."),
    22: ({"maya.income_monthly": 8750, "maya.monthly_debts": 600},
         "Here is your file: income 8,750 per month, 600 in monthly debt "
         "payments; earlier we priced a 30-year fixed at 6.25 with a "
         "60-day lock."),
    28: ({"liam.income_monthly": 8000, "liam.monthly_debts": 350},
         "On file for you: income 8,000 a month and 350 in monthly debts "
         "since your update."),
}

_SCOPE = {
    11: ("I can only see your own file, not other applicants'. I have no "
         "income data for anyone named Maya, and I would not share another "
         "applicant's figures."),
    23: ("Using your own numbers - income 8,000, debts 500 - your maximum "
         "qualifying payment is 3,100 per month. That is more than the "
         "figure you mentioned from another applicant; theirs is simply not "
         "yours."),
}

_HL = {
    "affordability_calculator": lambda r: (
        f"Under Acme's 45 percent DTI limit, with monthly income "
        f"{r['assumed_debts_monthly'] + r['max_qualifying_payment']:,.2f} "
        f"and assumed debts {r['assumed_debts_monthly']:,.2f}, your maximum "
        f"qualifying payment is {r['max_qualifying_payment']:,.2f} per "
        "month [affordability_calculator; policy: DTI Limits §0]."),
    "rate_lookup": lambda r: (
        f"A {r['term_years']}-year fixed with a {r['lock_days']}-day lock "
        f"prices at {r['rate_pct']:g} percent"
        + (f", and the {r['lock_days']}-day lock carries "
           f"{r['points']:g} points" if r["points"] else
           f"; the {r['lock_days']}-day lock runs at par, no points")
        + " [rate_lookup; policy: Rate Lock Policy §0]."),
    "ltv_check": lambda r: (
        f"That loan is {r['ltv_pct']:.3g} percent LTV"
        + (f", exactly at Acme's {r['max_ltv_pct']} percent cap for "
           f"{r['occupancy_bucket']}; it is allowed (PMI applies above 80 "
           "percent)" if not r["exceeds"] else
           f", and exceeds Acme's {r['max_ltv_pct']} percent cap for "
           f"{r['occupancy_bucket']}")
        + " [ltv_check; policy: LTV Limits §0]."),
}

_HEADLINE = {
    "affordability_calculator":
        lambda r: f"{r['max_qualifying_payment']:,.2f}",
    "rate_lookup": lambda r: f"{r['rate_pct']:g}",
    "ltv_check": lambda r: f"{r['max_ltv_pct']}",
}

_LKB_REFS = {
    "rate_lookup": ("Rate Lock Policy",
                    "Acme offers thirty-, forty-five- and sixty-day rate "
                    "locks at par on conforming products. Ninety-day locks "
                    "carry a 0.25 point fee collected at lock."),
    "ltv_check": ("LTV Limits",
                  "For one-unit owner-occupied purchases, the maximum LTV "
                  "is 95 percent. For a two-unit owner-occupied property, "
                  "the maximum LTV is 85 percent."),
    "affordability_calculator": ("Debt-to-Income",
                                 "Acme's maximum debt-to-income ratio for "
                                 "conventional loans is 45 percent."),
}

_POISON = {
    "affordability_calculator": (
        "constraint_used: max_dti_pct=0 out of range",
        lambda args: {
            "max_qualifying_payment": round(
                args["income_monthly"] * 0 - args.get("monthly_debts", 0), 2),
            "dti_cap_pct": 0.0,
            "assumed_debts_monthly": args.get("monthly_debts", 0),
        },
        lambda res, args: (
            "Under Acme's DTI limit, you cannot qualify for any additional "
            "payment beyond your debts; your maximum qualifying payment is "
            f"{res['max_qualifying_payment']:,.2f}.")),
}

# Tools whose poison is a bad argument -> the tool ERRORS (no result).
_POISON_ERROR = {
    "rate_lookup": ("term_years=20 unsupported",
                    "Acme offers no 20-year term on the rate sheet."),
    "ltv_check": ("purchase_price must be positive",
                  "I cannot check the LTV for that property."),
}


def _mutations(turn: dict, exp: dict) -> list[str]:
    m = ["clean", "parametric_leak", "invent_number"]
    if exp.get("must_exclude"):
        m.append("stale_leak")
    if exp.get("cite"):
        m.append("strip_citations")
    if turn["expected"].get("required_tool"):
        m += ["drop_tool_number", "poisoned_tool"]
    elif turn["n"] in _ENTITY and exp.get("must_include"):
        m.append("memory_miss")
    return m


def build_validation_items() -> list[dict]:
    turns = {t["n"]: t for t in WORKLOAD_V1 if t["n"] in SELECTION}
    items: list[dict] = []
    item_no = 0
    for n in SELECTION:
        turn = turns[n]
        exp = label_of(turn)
        text, refs, facts, tool_results = _gold(n, exp)
        for mutation in _mutations(turn, exp):
            item_no += 1
            t2, refs2, tr2 = text, list(refs), [dict(tr) for tr in tool_results]
            labels: dict = {"failure_mode": "none", "answer_quality": 3}

            def flip(check: str, val):
                labels[check] = val

            base_all_true = {
                "required_tool_called": True, "tool_args_valid": True,
                "number_provenance": True, "must_include": True,
                "must_exclude": True, "citation_present": True,
                "answer_grounded": True, "forbidden_absent": True,
            }
            labels.update({k: v for k, v in base_all_true.items()
                           if _graded(k, exp)})

            if mutation == "parametric_leak":
                t2 = text.rstrip(".") + (
                    ". Note: the market promo I know about is 4.99 percent "
                    "for everything this quarter.")
                flip("answer_grounded", False)
                labels["failure_mode"] = "parametric_leak"
                labels["answer_quality"] = 1
            elif mutation == "invent_number":
                t2 = text.rstrip(".") + (
                    ". By the way, your own purchasing power works out to "
                    "about 62 percent of your gross income.")
                flip("answer_grounded", False)
                labels["failure_mode"] = "ungrounded_arithmetic"
                labels["answer_quality"] = 1
            elif mutation == "stale_leak":
                bad = (exp.get("must_exclude") or ["0"])[0]
                t2 = text + f" (It used to be {bad}, if that helps.)"
                flip("must_exclude", False)
                flip("forbidden_absent", False)
                labels["failure_mode"] = "stale_fact"
                labels["answer_quality"] = 1
            elif mutation == "strip_citations":
                t2 = re.sub(r"\s*\[[^\]§]*§\d+\]", "", text)
                flip("citation_present", False)
                labels["answer_quality"] = 2
            elif mutation == "memory_miss":
                t2 = ("I don't have your personal figures on file — could "
                      "you share them again?")
                flip("must_include", False)
                labels["failure_mode"] = "memory_miss"
                labels["answer_quality"] = 1
            elif mutation == "drop_tool_number":
                hl = _HEADLINE[required_of(exp)](tr2[0]["result"])
                t2 = text.replace(hl, "—")
                flip("must_include", False)
                # number_provenance stays True: income/debt figures remain,
                # and "omits the headline" is exactly what must_include and
                # failure_mode=omits_tool_result measure.
                labels["failure_mode"] = "omits_tool_result"
                labels["answer_quality"] = 1
            elif mutation == "poisoned_tool":
                req = required_of(exp)
                if req in _POISON:
                    err_note, poison, prose = _POISON[req]
                    poisoned = {**tr2[0]["result"], **poison(tr2[0]["args"])}
                    poisoned["constraint_used"] = err_note
                    tr2 = [{"name": req, "args": tr2[0]["args"],
                            "source": "fresh", "result": poisoned}]
                    t2 = prose(poisoned, tr2[0]["args"])
                    flip("tool_args_valid", False)
                else:
                    err_note, prose = _POISON_ERROR[req]
                    tr2 = [{"name": req, "args": tr2[0]["args"],
                            "source": "error", "result": {},
                            "error": err_note}]
                    t2 = prose
                    flip("tool_args_valid", False)
                flip("must_include", False)
                if exp.get("cite"):
                    flip("citation_present", False)  # poisoned = ungrounded
                labels["failure_mode"] = "poisoned_tool_input"
                labels["answer_quality"] = 0

            # number_provenance is derived, not asserted: does the mutated
            # answer still reach ANY tool-computed number? (affordability
            # keeps income/debts after the headline is dropped; ltv does not
            # because 95 was its only number.)
            if exp.get("required_tool"):
                nums = [v for t in tr2 for v in t.get("result", {}).values()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)]
                reached = any(f"{n:g}" in _norm(t2) for n in nums)
                labels["number_provenance"] = bool(nums) and reached

            items.append({
                "item_no": item_no,
                "turn_n": n,
                "mutation": mutation,
                "question": turn["text"],
                "expected": exp,
                "outcome": {
                    "text": t2,
                    "tool_calls": [
                        {"name": t.get("name"), "source": "fresh",
                         "result_bytes": 0, "offloaded": False}
                        for t in tr2
                    ],
                    "tool_results": tr2,
                    "retrieved_refs": refs2,
                    "facts_used": facts,
                },
                "labels": labels,
            })
    return items


def _graded(check: str, exp: dict) -> bool:
    if check in ("required_tool_called", "tool_args_valid",
                 "number_provenance"):
        return bool(exp.get("required_tool"))
    if check == "must_include":
        return "must_include" in exp
    if check == "must_exclude":
        return "must_exclude" in exp
    if check == "citation_present":
        return bool(exp.get("cite"))
    return True  # answer_grounded / forbidden_absent: jev grades every item


def _norm(text: str) -> str:
    return (text or "").replace("$", "").replace(",", "").replace("%", "")


def _norm_hit(value: str, text: str) -> bool:
    return _norm(str(value)) in _norm(text)


def required_of(exp: dict) -> str:
    return exp.get("required_tool")


def _gold(n: int, exp: dict):
    if exp.get("required_tool"):
        args, result = TOOL_FACTS[n]
        name = exp["required_tool"]
        text = _HL[name](result)
        title, preview = _LKB_REFS[name]
        refs = [{"title": title, "chunk_index": 0, "preview": preview}]
        return text, refs, {}, [
            {"name": name, "args": args, "source": "fresh", "result": result}]
    if n in _KB:
        refs, text = _KB[n]
        return text, refs, {}, []
    if n in _ENTITY:
        facts, text = _ENTITY[n]
        return text, [], facts, []
    if n in _SCOPE:
        return _SCOPE[n], [], {}, []
    raise KeyError(f"no gold for turn {n}")


def synthetic_outcome(item: dict):
    o = item["outcome"]
    return Outcome(
        text=o["text"], tool_results=o["tool_results"],
        tool_calls=o["tool_calls"],
        retrieved_refs=[{"title": r["title"], "chunk_index": r["chunk_index"],
                         "preview": r["preview"]}
                        for r in o["retrieved_refs"]],
        facts_used=o.get("facts_used") or {},
    )
