# Evaluation

The evaluation harness grades every answer against constructed labels and reports
grader agreement (Cohen's κ), so the numbers are inspectable, not claimed.

## The workload

`WORKLOAD_V1` (32 turns, two users, seven scenarios) exercises one memory promise per
scenario:

| scenario | memory promise |
|---|---|
| supersession | a newer fact value replaces the older one, everywhere |
| cross-thread | a fact written in thread A is visible in thread B |
| tool | required-tool discipline and provenanced numbers |
| kb grounding | citations `[title §chunk]`, policy numbers only from retrieved text |
| scope | an intruder user never sees another user's facts |
| recall | paraphrase/compaction of earlier turns |
| cache | a repeated question is served with ~0 tokens |

## Grader cascade

- **Code** grades what code can prove deterministically: `required_tool_called`,
  `tool_args_valid`, `number_provenance`, `must_include`, `must_exclude`,
  `citation_present`.
- **Jev** (TypeSafe System One) grades judgment checks against the trace as a structured
  state in one batched call: `answer_grounded`, `forbidden_absent`, `failure_mode`,
  `answer_quality`.
- **Judge** (a configurable model, `MEM_JUDGE_MODEL`; it defaults to the chat model, so
  it is only "cross-family" when you point it at a different family) is the tie-breaker
  for low-confidence Jev answers — and is **gated off**, because with a fixed prompt and a
  fair (all-items) measurement it still never beat Jev on any check. Any cross-family
  claim must name both model IDs in a committed run artifact.

## Authority (pre-declared rule)

For each check, the authoritative grader is the one with the highest κ ≥ 0.6; ties break
to the cheaper grader. Measured on 96 constructed-label items:

| check | grader | κ / agreement |
|---|---|---|
| required_tool_called, tool_args_valid, number_provenance, must_include, must_exclude, citation_present | code | **1.0** |
| forbidden_absent | jev | **1.0** |
| failure_mode | jev | 0.83 exact |
| answer_grounded | jev | 0.79 |
| answer_quality | jev | 0.68 exact / **0.99 ±1** (soft signal) |
| (any) | judge | < Jev — disabled |

`answer_grounded` over-flags a few entity-recall items where the model re-formats a
number (`8,750` vs `8750`) — a documented limitation, still above the 0.6 bar.

## Cost

Jev pricing is anchored to real billing (not the per-billion-vs-per-million unit slip the
first version had): a full 96-item validation run costs **~$0.004**. The chat path stays
OpenAI (one variable at a time); Jev never sits in the runtime.

## Running it

```powershell
uv run python scripts/run_eval.py --backend memory,langgraph,baseline   # workload -> mem.eval_*
uv run python scripts/validate_graders.py --json eval_validation.json   # grader kappa table
```
