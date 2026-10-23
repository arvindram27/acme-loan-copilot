# Acme Loan Copilot — agent memory built twice

A v1 modeled on Part 7 of [RichmondAlake's agent_memory_course](https://github.com/RichmondAlake/agent_memory_course):
an agent memory system built **from scratch as SQL on Postgres**, then again **through
LangGraph's store**, and evaluated so the two designs can be compared on accuracy,
latency, tokens and cost. Domain: **Acme Home Loans**, a loan-officer copilot.

## Stack

- **Postgres 16 + pgvector** in Docker (two databases: `agentmem` = scratch side, `agentmem_lg` = LangGraph side)
- **Python 3.12** via `uv` (venv lives *outside* this OneDrive-synced folder)
- **OpenRouter** — one OpenAI-compatible key covers chat, embeddings and the cross-family judge
- **FastAPI** for the app pages (later milestones)

## Setup

```powershell
# 1. Database (Docker Desktop must be running)
cd infra
docker compose up -d

# 2. Python environment
cd ..
$env:UV_PROJECT_ENVIRONMENT = 'C:\Users\Dhanalakshmi\.venvs\agent-memory'
uv sync

# 3. Keys
Copy-Item .env.example .env      # then put an OpenRouter key in MEM_API_KEY

# 4. Migrations
uv run python -m agentmem.db
```

## Tests

```powershell
uv run pytest                    # contracts + real-database checks
uv run pytest -m live            # (live model tests skip without a key)
uv run python scripts/smoke_chat.py
```

Two Windows gotchas baked into the config:

- The venv path is set per-shell with `UV_PROJECT_ENVIRONMENT` so OneDrive never syncs it.
- DSNs use `127.0.0.1`, not `localhost` — Docker maps the port on IPv4 only here, and
  libpq's first attempt on `::1` hangs instead of failing fast.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 0 | Foundation: infra, migrations, adapters, contracts | **done** |
| 1 | Conversation + persona | **done** |
| 2 | Entity memory + supersession (`002_entity.sql`, live demo: corrected income resolves, stale excluded) | **done** |
| 2 | Entity memory + supersession | — |
| 3 | Knowledge base + hybrid retrieval | — |
| 4 | Toolbox + tool log | — |
| 5 | Working memory, summaries, semantic cache | — |
| 6 | Evaluation harness | — |
| 7 | LangGraph backend + comparison | — |
| 8 | App: chat, memory inspector, data explorer, evals | — |

See `C:\Users\Dhanalakshmi\.opencode\plan\agent-memory-loan-copilot.md` for the full plan.
