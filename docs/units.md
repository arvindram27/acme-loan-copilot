# Memory units

The copilot's memory is layered. Each unit answers three questions: **what it
stores**, **when to use it**, and **what it is *not* for**. Scope (`user_id` /
`thread_id`) is always set by the app, never by the model.

| # | Unit | Scratch table(s) | Read path | Demo moment |
|---|---|---|---|---|
| 0 | Stateless baseline | — | prompt only | "I don't know your applicant" |
| 1 | Conversation (episodic) | `mem.turn` | last N + vector recall in thread | back-reference to a co-borrower |
| 2 | Persona | `mem.persona` (versioned) | active version only | persona survives new threads |
| 3 | Entity memory | `mem.entity`, `mem.fact` | resolve current facts | income 95k → 105k, stale value excluded |
| 4 | Knowledge base | `mem.doc`, `mem.chunk` | top-k + min-score, chunk ids | cited max-LTV answer |
| 5 | Toolbox + tool log | `mem.tool`, `mem.tool_call` | tool choice by vector search; reuse fresh results; offload large payloads | repeated rate call = hit |
| 6 | Working memory + summaries | `mem.summary` | token-budgeted context builder | 30-turn interview compacts, cites turns |
| 7 | Semantic cache | `mem.cache` | similarity ≥ threshold AND same scope/epoch | repeat question ~0 tokens |

Every unit follows one contract: `memory_id, user_id, thread_id` (app-supplied),
`kind, content, embedding vector(1536), importance, status, supersedes_id, ttl_at,
source_ref`, plus a write path, a read path, and the "when to use / not for" note.

---

## Unit 0 — stateless baseline

No memory at all. Every request starts from nothing; the model can only answer from
its parametric knowledge. This is the contrast arm of the evaluation: it shows what
the memory system actually buys.

- **Not for:** anything real — it is the control, not a feature.

## Unit 1 — conversation (episodic memory)

One row per turn (`mem.turn`), scoped by `user_id`/`thread_id`, with an embedding of
the user's message so a *new* question can recall *old* similar turns even from far
back in a thread.

- **When to use it:** anything the agent or user said in this thread.
- **Not for:** durable facts (entity memory) or policy knowledge (the knowledge base).

## Unit 2 — persona

Who the agent should be for this user — style, strictness, rules they insist on.
Versioned: an update supersedes the active row and inserts a new version; exactly one
active version per user.

- **When to use it:** constant, thread-independent instructions for every request.
- **Not for:** facts that change per application (entity memory) or chat content.

## Unit 3 — entity memory (facts + supersession)

Durable facts about nouns — applicant, co-borrower, property, loan. **Append-only**:
a new value supersedes the old one in one transaction and keeps the full history, so a
corrected income never silently rewrites the record. A fact learned in one thread is
visible to later threads (that is what makes it *entity* memory).

- **When to use it:** values that outlive the conversation and span threads.
- **Not for:** conversation content, policy text, or behavior (persona).

## Unit 4 — knowledge base (hybrid retrieval)

The underwriting corpus, chunked, embedded at 1536-d, full-text indexed. Reads fuse
keyword (tsvector) and vector (cosine) ranks with reciprocal rank fusion, and enforce a
minimum vector score so unrelated chunks can't ride the fusion on rank position alone.
Answers cite `[title §chunk]`.

- **When to use it:** answers that must cite policy ("what's the max LTV for …").
- **Not for:** anything the user said. Policy text is **app-shared** across users by design.

## Unit 5 — toolbox + tool log

The deterministic tools that keep arithmetic in code (`affordability_calculator`,
`rate_lookup`, `ltv_check`, `policy_full_dump`). Tool *choice* is vector search over
descriptions; the *log* remembers what ran, keyed by an args hash, so an identical call
within the tool's TTL is a cache hit. Large results are offloaded to an artifact; the row
keeps a pointer + preview.

- **When to use it:** any computation — never let the model do arithmetic.
- **Not for:** storing facts; the tool log is provenance, not truth.

## Unit 6 — working memory (summaries)

When a thread outgrows the verbatim recency window, older turns fold into a compressed
running summary that cites the turn seqs it covers. Summaries are append-only versions and
are **fact-aware**: current entity facts are injected so a summary never re-asserts a
superseded value as current.

- **When to use it:** compressing a long thread so context stays in budget.
- **Not for:** durable facts — entity memory is authoritative; a summary is prose.

## Unit 7 — semantic answer cache

Memoizes a whole answer keyed by `(user_id, thread_id, fact-epoch, question embedding)`.
A hit requires the **same scope**, an **unchanged fact-epoch** (any fact write bumps it,
so a cached answer can never go stale), and embedding similarity ≥ threshold. `expires_at`
is an independent backstop for questions that age even when facts don't.

- **When to use it:** a repeat/paraphrase of an already-answered question.
- **Not for:** turns that must go through a tool, or that state new facts (both are gated out).
