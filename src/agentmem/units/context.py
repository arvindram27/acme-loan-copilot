"""Unit 6 — context assembly with a token budget.

Composes the system prompt from the memory units in a fixed priority order
(system -> persona -> facts -> summary -> recall -> policy excerpts) and
applies a soft token budget: when the estimate would exceed the ceiling, the
lowest-priority sections (recalled turns, then policy excerpts) are trimmed.
The verbatim recency window is already bounded by `recency_k`, so the budget
is a backstop against a pathological persona/facts/summary, not the primary
compression — that job belongs to working-memory summaries.

Returns the assembled system text, the verbatim messages, and a per-section
token accounting so the eval can report where the budget went and prove the
answer came from a summary rather than unbounded raw history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentmem.units import conversation, knowledge
from agentmem.units.working import format_summary

RECALL_HEADER = "## Earlier in this thread (semantically relevant)"
KB_HEADER = (
    "## Policy excerpts (cite as [title §chunk]; quote numbers only from here)"
)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (no tokenizer dependency): ~4 chars/token."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class Assembled:
    system: str
    messages: list[dict]              # verbatim recent turns, before user msg
    sections: dict[str, int]          # section name -> estimated tokens
    context_tokens: int               # system + messages estimate
    summary_used: bool
    recalled_turns: int
    chunks: list[dict]                # policy excerpts actually included


def assemble(
    *,
    system_base: str,
    persona_text: str,
    facts_text: str,
    summary: dict | None,
    recalled: list[dict],
    chunks: list[dict],
    recent_before_current: list[dict],
    max_context_tokens: int,
) -> Assembled:
    messages = [
        {"role": t["role"], "content": t["content"]}
        for t in recent_before_current
    ]
    msg_tokens = sum(estimate_tokens(m["content"]) for m in messages)

    # Fixed sections are always kept (they are small and load-bearing).
    parts: list[tuple[str, str]] = [("system", system_base)]
    if persona_text:
        parts.append(("persona", f"## Persona\n{persona_text}"))
    if facts_text:
        parts.append(("facts", f"## Known facts\n{facts_text}"))
    if summary:
        parts.append(("summary", format_summary(summary)))

    budget = max_context_tokens - msg_tokens - sum(
        estimate_tokens(t) for _, t in parts
    )

    # Trim lowest priority first: recall, then policy excerpts.
    kept_recall: list[dict] = []
    kept_chunks: list[dict] = []
    for t in recalled:
        cost = estimate_tokens(t["role"]) + estimate_tokens(t["content"]) + 4
        if budget - cost < 0:
            break
        kept_recall.append(t)
        budget -= cost
    for c in chunks:
        cost = estimate_tokens(c["title"]) + estimate_tokens(c["text"]) + 8
        if budget - cost < 0:
            break
        kept_chunks.append(c)
        budget -= cost

    if kept_recall:
        parts.append(
            ("recall", RECALL_HEADER + "\n" + conversation.format_turns(kept_recall))
        )
    if kept_chunks:
        parts.append(("kb", KB_HEADER + "\n" + knowledge.format_chunks(kept_chunks)))

    system = "\n\n".join(t for _, t in parts)
    sections = {n: estimate_tokens(t) for n, t in parts}
    return Assembled(
        system=system,
        messages=messages,
        sections=sections,
        context_tokens=sum(sections.values()) + msg_tokens,
        summary_used=summary is not None,
        recalled_turns=len(kept_recall),
        chunks=kept_chunks,
    )
