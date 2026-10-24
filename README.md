# Acme Loan Copilot — agent memory built twice

A v1 modeled on Part 7 of [RichmondAlake's agent_memory_course](https://github.com/RichmondAlake/agent_memory_course):
an agent-memory system built **from scratch as SQL on Postgres**, then again **through
LangGraph's store and checkpointer**, and evaluated so the two designs can be compared on
accuracy, latency, tokens and cost. Domain: **Acme Home Loans**, a loan-officer copilot.

The copilot remembers the applicant, corrects itself when facts change (supersession), never
leaks one applicant's data to another (scope), grounds policy answers in a cited corpus, and
keeps arithmetic inside deterministic tools — never in the model's head.

## Stack

- **Postgres 16 + pgvector** in Docker — two databases: `agentmem` (scratch) and `agentmem_lg` (LangGraph)
- **Python 3.12** via `uv` (venv lives *outside* the OneDrive-synced folder)
- **OpenRouter** — one OpenAI-compatible key covers chat and embeddings; the eval judge is a configurable model (`MEM_JUDGE_MODEL`), currently gated off
- **LangGraph** `PostgresStore` + `PostgresSaver` for the second backend
- **FastAPI** for the app pages

## Setup

```powershell
# 1. Database (Docker Desktop must be running)
cd infra
docker compose up -d

# 2. Python environment (venv outside OneDrive)
cd ..
$env:UV_PROJECT_ENVIRONMENT = 'C:\Users\Dhanalakshmi\.venvs\agent-memory'
uv sync

# 3. Keys
Copy-Item .env.example .env   # put an OpenRouter key in MEM_API_KEY (optional MEM_JEV_KEY)

# 4. Migrations + corpus
uv run python -m agentmem.db
uv run python scripts/seed_kb.py
```

Two Windows gotchas baked into the config:

- The venv path is set per-shell with `UV_PROJECT_ENVIRONMENT` so OneDrive never syncs it.
- DSNs use `127.0.0.1`, not `localhost` — Docker maps the port on IPv4 only here, and
  libpq's first attempt on `::1` hangs instead of failing fast.

## Run it

```powershell
# Web app (chat / memory inspector / data explorer / evals)
uv run python scripts/serve.py --port 8730     # then open http://127.0.0.1:8730

# One-shot chat smoke test
uv run python scripts/smoke_chat.py

# Demos (each exercises one memory unit)
uv run python scripts/demo_baseline.py
uv run python scripts/demo_kb.py
uv run python scripts/demo_tools.py
```

## Evaluate it

```powershell
# Golden workload through one or more backends (persists to mem.eval_*)
uv run python scripts/run_eval.py --backend memory,langgraph,baseline

# Validate the graders against constructed labels (prints the κ table)
uv run python scripts/validate_graders.py

# Five very different questions, on both backends, with assertions (exit 0 = all pass)
uv run python scripts/demo_check.py
```

### Results (32-turn workload)

| backend | accuracy | op_score | p50 | note |
|---|---|---|---|---|
| memory (scratch) | **1.00** | **1.00** | ~4.1s | + summaries + semantic answer cache |
| langgraph | **1.00** | **1.00** | ~3.6s | equivalent on every memory promise |
| baseline | 0.25 | 0.14 | ~1.1s | no memory at all |

Grader authority (pre-declared rule: highest κ ≥ 0.6 wins per check): **6 deterministic
checks → code** (κ = 1.0); **3 judgment checks → Jev** (`forbidden_absent` 1.0,
`answer_grounded` 0.79, `failure_mode` 0.83); `answer_quality` is a soft signal (±1 = 0.99);
the LLM judge is gated off because it never beat Jev. A full Jev validation run costs ~$0.004.

## Screenshots

The Memory Ladder app (run `uv run python scripts/serve.py`, then open http://127.0.0.1:8730):

<img src="docs/screenshots/01-overview.png" alt="Overview — the memory ladder" width="720">

<img src="docs/screenshots/02-chat-scratch.png" alt="Chat — scratch backend with the live memory panel" width="720">

<img src="docs/screenshots/03-chat-langgraph.png" alt="Chat — LangGraph backend (the backend switcher)" width="720">

<img src="docs/screenshots/04-entity-supersession.png" alt="Entity rung — superseded facts struck through" width="720">

<img src="docs/screenshots/05-measure.png" alt="Measure — evaluation runs" width="720">

<img src="docs/screenshots/06-data.png" alt="Data explorer" width="720">

## Tests

```powershell
uv run pytest                    # 97 contracts + real-database checks
uv run pytest -m live            # live model tests (skip without a key)
```

## Milestones

| # | Milestone | Status |
|---|---|---|
| 0 | Foundation: infra, migrations, adapters, contracts | **done** |
| 1 | Conversation + persona | **done** |
| 2 | Entity memory + supersession (corrected income resolves, stale excluded) | **done** |
| 3 | Knowledge base + hybrid RRF retrieval + cited answers + guarded extraction | **done** |
| 4 | Toolbox + tool log (semantic tool choice, args-hash cache, artifact offload, planner loop) | **done** |
| 5 | Working memory + summaries + semantic answer cache | **done** |
| 6 | Evaluation harness (constructed-label validation, code→Jev→judge cascade, κ table) | **done** |
| 7 | LangGraph backend + comparison | **done** |
| 8 | App: chat, memory inspector, data explorer, evals | **done** |
| 9 | Docs: unit pages, diagrams, README | **done** |

## Docs

- [Units](docs/units.md) — what each memory unit stores, when to use it, what it is *not* for
- [Architecture](docs/architecture.md) — schema, context assembly, and grader-cascade diagrams
- [Backends](docs/backends.md) — scratch SQL vs LangGraph, side by side
- [Evaluation](docs/evaluation.md) — the harness, grader authority, and results

See `C:\Users\Dhanalakshmi\.opencode\plan\agent-memory-loan-copilot.md` for the full plan.
