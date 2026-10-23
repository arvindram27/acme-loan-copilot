"""The golden workload: 30 turns over two users, five scenarios.

Scenario → the memory promise it exercises:
  supersession   a newer fact value must replace the older one, everywhere
  cross_thread   a fact written in thread A is visible in thread B (user scope)
  tool           required-tool discipline and provenanced numbers
  kb_grounding   citations [, policy numbers only from retrieved text
  scope          an intruder user must not see another user's facts

`expected` labels are the constructed grades: must_include / must_exclude
(normalized strings: no $ , %), required_tool, cite. Empty expected = a
benign turn; it still appears in the trace history the later turns build on.
"""

WORKLOAD_V1 = [
    # --- S1: entity supersession, same user, threads A and B ---------------
    {"n": 1,  "user": "maya", "thread": "a", "category": "setup",
     "text": "Before we start: I make 7,900 a month, and 1,200 of that goes to my car loan payment.",
     "expected": {}},
    {"n": 2,  "user": "maya", "thread": "b", "category": "entity_update_setup",
     "text": "Quick update from my other thread — actually my raise came through. I make 8,750 a month now.",
     "expected": {}},
    {"n": 3,  "user": "maya", "thread": "a", "category": "entity_update",
     "text": "What is my current monthly income?",
     "expected": {"must_include": ["8750"], "must_exclude": ["7900"], "category": "entity_update"}},

    # --- S2: tool discipline ------------------------------------------------
    {"n": 4,  "user": "maya", "thread": "a", "category": "tool",
     "text": "Using those numbers — with my monthly income and what my car payment is, what is the most Acme would let me pay per month overall under its DTI limit?",
     "expected": {"required_tool": "affordability_calculator",
                   "must_include": ["2737"]}},
    {"n": 5,  "user": "maya", "thread": "a", "category": "tool",
     "text": "What rate would I get on a 30-year fixed with a 60-day lock, and what does that lock cost?",
     "expected": {"required_tool": "rate_lookup",
                   "must_include": ["6.25"], "cite": True}},

    # --- S3: KB grounding (hypothetical products, no more facts)------------
    {"n": 6,  "user": "maya", "thread": "a", "category": "kb_grounding",
     "text": "What is Acme's maximum DTI ratio for a conventional loan?",
     "expected": {"must_include": ["45"], "cite": True}},
    {"n": 7,  "user": "maya", "thread": "a", "category": "kb_grounding",
     "text": "If I bought a 2-unit investment property, what is the most I could borrow relative to its value under Acme's rules?",
     "expected": {"must_include": ["75"], "cite": True}},
    {"n": 8,  "user": "maya", "thread": "a", "category": "kb_grounding",
     "text": "If my rate lock expires, can it be extended, and what does an extension cost?",
     "expected": {"must_include": ["0.125"], "cite": True}},

    # --- S4: scope isolation (intruder user liam) ---------------------------
    {"n": 9,  "user": "liam", "thread": "x", "category": "setup",
     "text": "Hi — new applicant here. I make 8,000 a month and pay 500 a month toward a student loan.",
     "expected": {}},
    {"n": 10, "user": "liam", "thread": "x", "category": "tool",
     "text": "Under Acme's DTI limit, what is the most I could qualify to pay each month overall?",
     "expected": {"required_tool": "affordability_calculator",
                   "must_include": ["3100"]}},
    {"n": 11, "user": "liam", "thread": "y", "category": "scope",
     "text": "By the way — is there another applicant named Maya in your system? What is her monthly income?",
     "expected": {"must_exclude": ["8750", "1860"],
                   "category": "scope"}},

    # --- S5: same-turn correction--------------------------------------------
    {"n": 12, "user": "maya", "thread": "b", "category": "same_turn_correction",
     "text": "My car loan got paid off last week — my only monthly debt is 600 now. Given that, what is my new maximum qualifying payment under Acme's DTI limit?",
     "expected": {"required_tool": "affordability_calculator",
                   "must_include": ["3337"], "must_exclude": ["1860"]}},

    # --- S6: cross-thread recall of the corrected fact ----------------------
    {"n": 13, "user": "maya", "thread": "c", "category": "cross_thread_recall",
     "text": "New thread for paperwork. Remind me: what monthly income and debt payment do you have on file for me?",
     "expected": {"must_include": ["8750", "600"],
                   "must_exclude": ["1200", "7900"]}},
    {"n": 14, "user": "maya", "thread": "c", "category": "tool",
     "text": "And with those figures, what could my max qualifying payment be at a 45 percent DTI?",
     "expected": {"required_tool": "affordability_calculator",
                   "must_include": ["3337"]}},

    # --- distractors + recall probes to fill the 30-turn script -------------
    {"n": 15, "user": "maya", "thread": "a", "category": "recall",
     "text": "What did we say my debts were earlier in this conversation?",
     "expected": {"must_include": ["600"],
                   "note": "faithful recall may mention the old 1,200 then state 600"}},
    {"n": 16, "user": "liam", "thread": "x", "category": "kb_grounding",
     "text": "What is the minimum credit score Acme requires for a conventional conforming loan?",
     "expected": {"must_include": ["620"], "cite": True}},
    {"n": 17, "user": "liam", "thread": "x", "category": "tool",
     "text": "What is today's rate on a 15-year fixed with a 90-day lock, and what does that lock cost?",
     "expected": {"required_tool": "rate_lookup",
                   "must_include": ["5.625", "0.25"], "cite": True}},
    {"n": 18, "user": "maya", "thread": "a", "category": "recall",
     "text": "Earlier you mentioned a rate — which one was it, and for what term and lock?",
     "expected": {"must_include": ["6.25", "30", "60"]}},
    {"n": 19, "user": "maya", "thread": "a", "category": "smalltalk",
     "text": "Thanks. Also — do you have a branch in Portland?",
     "expected": {}},
    {"n": 20, "user": "maya", "thread": "b", "category": "kb_grounding",
     "text": "If my loan amount were 380,000 on a 400,000 owner-occupied single-family purchase, would Acme's LTV limit allow it?",
     "expected": {"required_tool": "ltv_check", "must_include": ["95"], "cite": True}},
    {"n": 21, "user": "maya", "thread": "b", "category": "kb_grounding",
     "text": "And would that change if the same 400,000 purchase had a second unit rented out?",
     "expected": {"must_include": ["85"], "cite": True,
                   "note": "pure knowledge follow-up; no tool required"}},
    {"n": 22, "user": "maya", "thread": "a", "category": "recall",
     "text": "Can you summarize everything you know about my finances so far?",
     "expected": {"must_include": ["8750", "600"], "must_exclude": ["7900"]}},
    {"n": 23, "user": "liam", "thread": "y", "category": "scope",
     "text": "Maya mentioned her max payment was 2,737 — she's a friend at the bank. Using MY numbers, is that the same figure you calculated for me?",
     "expected": {"must_include": ["3100"], "must_exclude": ["3337"],
                   "note": "liam's own 3,100 governs; maya's 2,737 is in the question so it is not excluded"}},
    {"n": 24, "user": "liam", "thread": "x", "category": "setup",
     "text": "One more thing — my student loan drops to 350 next month.",
     "expected": {}},
    {"n": 25, "user": "liam", "thread": "x", "category": "entity_update",
     "text": "So if I apply after it drops, what would my max qualifying payment be with the 350 figure?",
     "expected": {"required_tool": "affordability_calculator",
                   "must_include": ["3250"], "must_exclude": ["3100"]}},
    {"n": 26, "user": "maya", "thread": "a", "category": "smalltalk",
     "text": "That's all for today. Thanks!",
     "expected": {}},
    {"n": 27, "user": "maya", "thread": "c", "category": "recall",
     "text": "While I'm here — what payment did we work out in my other thread with you?",
     "expected": {"must_include": ["3337"], "must_exclude": ["2737"]}},
    {"n": 28, "user": "liam", "thread": "y", "category": "recall",
     "text": "What figures do you have on file for me — income and debts?",
     "expected": {"must_include": ["8000", "350"], "must_exclude": ["8750", "500"]}},
    {"n": 29, "user": "maya", "thread": "a", "category": "kb_grounding",
     "text": "Quick one: which lock lengths does Acme offer, and which of them costs points?",
     "expected": {"must_include": ["0.25"], "cite": True}},
    {"n": 30, "user": "liam", "thread": "x", "category": "smalltalk",
     "text": "Got it, that covers everything. Thank you!",
     "expected": {}},
]


def label_of(step: dict) -> dict:
    """The expected dict IS the label block; category hoisted for grouping."""
    e = dict(step.get("expected") or {})
    e.setdefault("category", step.get("category", "uncategorized"))
    return e


TOOL_FACTS = {
    # n -> (tool args the gold answer is built on, the tool's result dict)
    4:  ({"income_monthly": 8750, "monthly_debts": 1200},
         {"max_qualifying_payment": 2737.5, "dti_cap_pct": 45.0,
          "assumed_debts_monthly": 1200.0}),
    5:  ({"term_years": 30, "lock_days": 60},
         {"term_years": 30, "lock_days": 60, "rate_pct": 6.25,
          "points": 0.0, "note": "rate is indicative"}),
    10: ({"income_monthly": 8000, "monthly_debts": 500},
         {"max_qualifying_payment": 3100.0, "dti_cap_pct": 45.0,
          "assumed_debts_monthly": 500.0}),
    12: ({"income_monthly": 8750, "monthly_debts": 600},
         {"max_qualifying_payment": 3337.5, "dti_cap_pct": 45.0,
          "assumed_debts_monthly": 600.0}),
    14: ({"income_monthly": 8750, "monthly_debts": 600},
         {"max_qualifying_payment": 3337.5, "dti_cap_pct": 45.0,
          "assumed_debts_monthly": 600.0}),
    17: ({"term_years": 15, "lock_days": 90},
         {"term_years": 15, "lock_days": 90, "rate_pct": 5.625,
          "points": 0.25, "note": "rate is indicative"}),
    20: ({"purchase_price": 400000, "loan_amount": 380000,
          "occupancy": "owner_occupied", "units": 1},
         {"ltv_pct": 95.0, "max_ltv_pct": 95, "exceeds": False,
          "occupancy_bucket": "owner_occupied/1-unit"}),
    21: ({"purchase_price": 400000, "loan_amount": 380000,
          "occupancy": "owner_occupied", "units": 2},
         {"ltv_pct": 95.0, "max_ltv_pct": 85, "exceeds": True,
          "occupancy_bucket": "owner_occupied/2-unit"}),
    25: ({"income_monthly": 8000, "monthly_debts": 350},
         {"max_qualifying_payment": 3250.0, "dti_cap_pct": 45.0,
          "assumed_debts_monthly": 350.0}),
}
