"""Jev smoke test: proves the key, lists models, and evaluates one tiny
batched request (two nouls + one choice). Costs a fraction of a cent."""

import sys

from agentmem.providers.jev import JevClient, JevError
from agentmem.settings import get_settings


def main() -> int:
    s = get_settings()
    jc = JevClient(s)
    if not jc.has_key:
        print("MEM_JEV_KEY not set — add it to agent-memory/.env")
        return 1
    try:
        models = jc.models()
    except JevError as exc:
        print(f"key rejected: {exc}")
        return 1
    print(f"auth OK — {len(models)} models: {[m for m in models if 'jev' in str(m)][:5]} ...")

    history = [
        {"role": "user", "content": "I make 6,800/mo, debts 1,200."},
        {"role": "assistant", "content": "Your max qualifying payment is $1,860."},
    ]
    r = jc.evaluate(
        state={
            "question": "What is the most I can qualify to pay per month?",
            "answer": history[1]["content"],
            "tool_calls": [{"name": "affordability_calculator",
                            "result": {"max_qualifying_payment": 1860.0}}],
            "trace": history,
        },
        questions={
            "numeric_provenance": {
                "type": "noul",
                "instructions": (
                    "Every numerical result in `answer` appears in "
                    "`tool_calls[*].result` or in `question`."
                ),
            },
            "answer_not_empty": {
                "type": "noul",
                "instructions": "Is `answer` a non-empty, coherent sentence?",
            },
            "tool_used": {
                "type": "choice",
                "instructions": "How many entries does `tool_calls` contain?",
                "criteria": {
                    "zero": "No tool calls at all.",
                    "one": "Exactly one tool call.",
                    "many": "Two or more tool calls.",
                },
            },
        },
    )
    print(f"model={r.model} latency={r.latency_ms}ms "
          f"tokens={r.input_tokens}/{r.output_tokens} cost=${r.cost_usd:.6f}")
    for qid, a in r.answers.items():
        conf = f" conf={a.confidence:.2f}" if a.confidence is not None else ""
        print(f"  {qid}: {a.type}={a.value!r}{conf}")
    if jc.misses:
        print(f"MISSING ANSWERS: {jc.misses}")
        return 1
    ok = (len(r.answers) == 3
          and r.answers["numeric_provenance"].value > 0.5
          and r.answers["tool_used"].value == "one")
    print("RESULT:", "PASS" if ok else "UNEXPECTED — check shapes above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
