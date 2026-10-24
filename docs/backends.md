# Backends — scratch SQL vs LangGraph

The same memory semantics, built twice. The seam is a structural contract
(`backends/base.py`): both backends expose `answer(*, thread_id, text,
entity_refs) -> TurnOutcome`, so one workload runs through both and the
difference is measured, not claimed.

| Memory unit | Scratch (`agentmem`) | LangGraph (`agentmem_lg`) |
|---|---|---|
| conversation (episodic) | `mem.turn` + vector column | `PostgresSaver` checkpointer |
| persona | `mem.persona` (versioned, append-only) | `PostgresStore` `(user, "persona")` |
| entity facts | `mem.fact` (append-only, supersession history kept) | `PostgresStore` `(user, "entity")`, `put()` overwrite |
| cross-thread visibility | user-scoped query | user-scoped namespace (automatic) |
| semantic recall | `ORDER BY embedding <=>` | `PostgresStore` `(user, "turns", thread)` + `search()` |
| knowledge base | shared `mem.chunk` | shared (app-shared by design) |
| tools + tool log | shared `toolbox` | shared |
| fact extraction | shared `extract` | shared |
| context assembly | shared `context` | shared |
| working-memory summaries | `mem.summary` | not ported yet (documented gap) |
| semantic answer cache | `mem.cache` | not ported yet (documented gap) |

## What's shared, and why

Only the *memory store* changes between the two backends. KB retrieval, fact
extraction, the deterministic toolbox, and context assembly are **application
logic**, so both backends drive them unchanged — the "one variable at a time"
discipline.

## Behavioral differences to know about

- **Fact history.** Scratch keeps append-only `mem.fact` rows (auditable: what the
  agent believed, and when). LangGraph's `put()` overwrites — current-facts-only.
  Correctness is identical; auditability differs.
- **Summaries + answer cache** are scratch-only for now. They re-implement cleanly on
  the store (summaries over checkpointer messages, cache keyed by the store fact-epoch)
  but were scoped out of milestone 7.

## Result (32-turn workload)

| backend | accuracy | op_score | p50 | notes |
|---|---|---|---|---|
| scratch | **1.00** | **1.00** | ~4.1s | summaries + answer cache (n=32 served with 0 tokens) |
| langgraph | **1.00** | **1.00** | ~3.6s | no summarization LLM calls |
| baseline | 0.25 | 0.14 | ~1.1s | control |

Both backends are behaviorally equivalent on every memory promise the workload
exercises — supersession, cross-thread recall, scope isolation, tool discipline,
KB grounding, and recall.

## LangGraph overwrite semantics are a choice, not a constraint

The LangGraph backend stores one entity document and rewrites it with `put()`
(a read-modify-write, no version chain). That is **this application's** design,
not something the store API forces — history and concurrency safety can both be
built on the same store (put a versioned document; compare-and-swap a version
field).

The trade-off to be able to state plainly: scratch keeps one row per fact
version with an at-most-one-active constraint (strong for corrected facts, with
an audit trail); the LangGraph backend takes the simpler document model
(simpler, weaker history, no conflict handling). To make it concurrency-safe:
CAS on a version field with bounded retries, or serialize writes per entity —
the same lock strategy scratch now uses.

Scratch's own concurrency story: `write_fact` takes a transaction-scoped
advisory lock per entity, so two concurrent corrections cannot both supersede
the same row (tested in `tests/test_robustness.py`).
