"""Validate the graders against constructed labels (Part 7 part 2).

Builds ~120 items with known grades by defect injection, grades each with
the code grader offline, and — when MEM_JEV_KEY is set — with Jev, plus
judge escalation for unsure Jev answers. Prints the k/agreement table per
(check, grader), plus cost and escalation stats.

Usage:
    python scripts/validate_graders.py [--json out.json]

Pre-declared selection rule (fixed before looking at results): for each
check, the authoritative grader is the one with the highest k provided
k >= 0.6; ties break to the cheaper grader. A check whose best k < 0.6
stays unvalidated and always escalates in production runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from agentmem.eval.graders import (
    CodeGrader, JevGrader, LLMJudge, run_cascade, Grade,
)
from agentmem.eval.labels import build_validation_items, synthetic_outcome
from agentmem.eval.metrics import grader_agreement
from agentmem.providers.chat import ChatClient
from agentmem.providers.jev import JevClient
from agentmem.settings import get_settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    s = get_settings()
    items = build_validation_items()
    print(f"constructed items: {len(items)}")

    jev_g = judge_g = None
    if s.jev_key:
        jev_g = JevGrader(JevClient(s), s.jev_escalate_low,
                          s.jev_escalate_high)
        judge_g = LLMJudge(ChatClient())
        print(f"Jev grader ON ({s.jev_model}); judge escalation enabled")
    else:
        print("MEM_JEV_KEY not set — code grader only (kappa vs labels)")

    rows: list[dict] = []
    escalated = defaultdict(int)
    for it in items:
        outcome = synthetic_outcome(it)
        grades = run_cascade(
            it["item_no"], it["expected"], outcome, it["question"],
            jev=jev_g, judge=judge_g,
        )
        for g in grades:
            rows.append({
                "item_no": it["item_no"], "mutation": it["mutation"],
                "check": g.check, "grader": g.grader,
                "pass_": None if g.pass_ is None else bool(g.pass_),
                "value": g.value if isinstance(g.value, (int, float, str))
                         else None,
                "probability": g.probability, "escalated": g.escalated,
                "tokens": g.tokens,
            })

    # ---- binary-check kappa table ------------------------------------
    # labels keyed like grader_agreement expects
    label_map: dict[str, dict[int, bool]] = defaultdict(dict)
    for it in items:
        for check, val in it["labels"].items():
            if isinstance(val, bool):
                label_map[check][it["item_no"]] = val
    table = grader_agreement(rows, dict(label_map))

    print("\n== kappa vs constructed labels (binary checks) ==")
    print(f"{'check|grader':<42} {'k':>7} {'agree':>7} {'n':>5}  conf")
    for key in sorted(table):
        r = table[key]
        conf = (f"tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}")
        print(f"{key:<42} {str(r['kappa']):>7} {str(r['agreement']):>7} "
              f"{r['n']:>5}  {conf}")

    # ---- choice/score checks: exact-match accuracy ---------------------
    print("\n== categorical checks (exact match) ==")
    for check in ("failure_mode", "answer_quality"):
        for grader in ("jev", "judge"):
            got = [(it["item_no"], r) for it, r in zip(items, rows)
                   if False]
    cat = defaultdict(lambda: [0, 0])
    by_item = defaultdict(dict)
    for r in rows:
        if r["pass_"] is None and r["grader"] in ("jev", "judge"):
            by_item[(r["grader"], r["check"])][r["item_no"]] = r["value"]
    for (grader, check), vals in by_item.items():
        ok = tot = 0
        for it in items:
            v = vals.get(it["item_no"])
            if v is None:
                continue
            lab = it["labels"][check]
            tot += 1
            if check == "answer_quality":
                ok += int(abs(float(v) - lab) <= 0.5)
            else:
                ok += int(v == lab)
        cat[(check, grader)] = [ok, tot]
    for (check, grader), (ok, tot) in sorted(cat.items()):
        print(f"{check}|{grader:<36} {ok}/{tot} = "
              f"{(ok / tot if tot else float('nan')):.3f}")

    if jev_g is not None:
        print(f"\njev calls: {jev_g.jev.calls}, "
              f"tokens {jev_g.jev.usage}, "
              f"cost ${jev_g.jev.total_cost_usd:.4f}")
        esc_rows = [r for r in rows if r["grader"] == "jev"
                    and r["escalated"]]
        total_jev = [r for r in rows if r["grader"] == "jev"]
        if total_jev:
            print(f"escalated: {len(esc_rows)}/{len(total_jev)} "
                  f"= {len(esc_rows) / len(total_jev):.1%}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "kappa": table}, f, default=str)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
