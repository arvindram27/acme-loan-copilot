"""Agent core: the stateless baseline and the memory-wired agent.

MemoryAgent's context assembly, in order (composed in units/context.py):
  1. persona (one active version per user, injected once)
  2. entity facts (current values only — superseded rows never leak)
  3. working-memory summary (older turns folded, cited by turn seq)
  4. recency (the last k turns, as proper chat messages)
  5. semantic recall of older turns (labelled, so provenance is visible)
  6. policy excerpts (hybrid retrieval, budget-trimmed last)

A paraphrase of an already-answered question (same scope, unchanged facts)
is served from the semantic answer cache with no completion. Every answer
persists both turns; the user turn is embedded for future recall. The
assistant turn is not embedded (halves embedding calls; recall targets what
the user asked, not what we replied).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from agentmem.eval.numcompare import contains_value
from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.units import (
    cache,
    context,
    conversation,
    entity,
    extract,
    knowledge,
    persona,
    toolbox,
    working,
)

SYSTEM_BASE = (
    "You are the loan-officer copilot for Acme Home Loans. Answer from the "
    "context you are given.\n"
    "- 'Known facts' are the applicant's current, authoritative record. If "
    "an older message in the conversation contradicts a known fact, the "
    "known fact wins (facts are superseded, the chat is not).\n"
    "- If a fact about an applicant is not in your context, say you don't "
    "know it.\n"
    "- Quote policy numbers only when they appear in retrieved knowledge; "
    "never invent rates. When your answer uses a policy rule (DTI, LTV, "
    "credit minimums, lock fees), cite it as [title §chunk] even when a "
    "tool also supplied the number."
)


_RECALL_WORDS = ("earlier", "before", "previously", "originally",
                 "used to", "at the start")


def _stale_hits(text: str, pairs: list[dict]) -> list[dict]:
    """Superseded values stated as current: old present, current absent.

    A value is only a hit when the draft mentions the superseded value AND omits
    the active one — so a faithful recall that mentions the old value and then
    states the current one is left alone.
    """
    hits = []
    for p in pairs:
        if (contains_value(text, p["old"])
                and not contains_value(text, p["current"])):
            hits.append(p)
    return hits


def _missing_numbers(answer: str, outcomes: list[dict], question: str) -> list[str]:
    """Numbers the tool computed that appear neither in the question nor in
    the draft answer. Explanation: arithmetic must come from tool output;
    a source number present in the question is excluded."""
    clean = answer.replace(",", "").replace("$", "").replace("%", "")
    qclean = question.replace(",", "").replace("$", "").replace("%", "")
    missing: set[str] = set()
    for t in outcomes:
        try:
            res = json.loads(t["result_for_model"])
        except json.JSONDecodeError:
            continue
        for v in res.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                num = f"{v:g}"
                if num in qclean:      # a number the user already said
                    continue
                if num not in clean:
                    missing.add(num)
    return missing


@dataclass
class TurnOutcome:
    text: str
    thread_id: str
    recent_turns: int = 0
    recalled_turns: int = 0
    facts_used: dict = field(default_factory=dict)
    facts_written: int = 0
    dropped_facts: list = field(default_factory=list)
    retrieved_refs: list = field(default_factory=list)   # [{title, chunk_index}]
    tool_calls: list = field(default_factory=list)       # name/source/bytes meta
    tool_results: list = field(default_factory=list)     # full results, for grading
    tokens: int = 0
    prompt_tokens: int = 0            # real prompt tokens of the final completion
    answer_cache_hit: bool = False    # served from mem.cache (no completion)
    context_tokens: int = 0           # estimated prompt tokens (system+messages)
    summary_used: bool = False        # a working-memory summary was in context
    summary_turns: int = 0            # turns folded into the summary


class BaselineAgent:
    """Unit 0: no memory at all. Every request starts from nothing."""

    def __init__(self, chat: ChatClient, system: str = SYSTEM_BASE):
        self.chat = chat
        self.system = system

    def answer(self, messages: list[dict]) -> str:
        out = self.chat.complete(
            [{"role": "system", "content": self.system}, *messages],
        )
        return out.text

    def say(self, text: str) -> str:
        return self.answer([{"role": "user", "content": text}])


class MemoryAgent:
    """Units 1-3 wired together: conversation, persona, entity memory."""

    def __init__(
        self,
        conn,
        *,
        user_id: str,
        chat: ChatClient,
        embedder: Embedder,
        recency_k: int = 10,
        recall_k: int = 4,
        kb_min_score: float = 0.30,
        extract_turns: bool = True,
        use_tools: bool = True,
        tool_rounds: int = 3,
        use_cache: bool = True,
        cache_min_score: float = 0.62,
        cache_ttl: int | None = None,
        max_context_tokens: int = 2800,
    ):
        self.conn = conn
        self.user_id = user_id
        self.chat = chat
        self.embedder = embedder
        self.recency_k = recency_k
        self.recall_k = recall_k
        self.kb_min_score = kb_min_score
        self.extract_turns = extract_turns
        self.use_tools = use_tools
        self.tool_rounds = tool_rounds
        self.use_cache = use_cache
        self.cache_min_score = cache_min_score
        self.cache_ttl = cache_ttl
        self.max_context_tokens = max_context_tokens
        persona.ensure_default(conn, user_id)
        # The planner needs exact param names; registry embeddings must exist.
        toolbox.sync_tools(conn, embedder)

    def answer(
        self,
        *,
        thread_id: str,
        text: str,
        entity_refs: list[tuple[str, str]] | None = None,
    ) -> TurnOutcome:
        """One user turn in, one assistant turn out; memory read and written."""
        conn = self.conn
        query_vec = self.embedder.embed(text)

        # The turn is stored first (so recall can step behind it), but not
        # shown in the recency window twice: the window is the turns BEFORE
        # this one, and this one rides as the user message.
        seq = conversation.append_turn(
            conn, user_id=self.user_id, thread_id=thread_id,
            role="user", content=text, embedding=query_vec,
        )["seq"]

        # Semantic answer cache (unit 6): a paraphrase of a question we have
        # already answered, with unchanged facts, is served verbatim — no
        # extraction, no tool loop, no completion. Gated to interrogative
        # turns that do not require a tool, so fact statements are never
        # swallowed and the eval's required-tool discipline is never skipped.
        cacheable = (
            self.use_cache
            and toolbox.required_tool(text) is None
            and cache.is_question(text)
        )
        if cacheable:
            hit = cache.lookup(
                conn, user_id=self.user_id, thread_id=thread_id,
                query_embedding=query_vec, min_score=self.cache_min_score,
            )
            if hit is not None:
                conversation.append_turn(
                    conn, user_id=self.user_id, thread_id=thread_id,
                    role="assistant", content=hit["answer"],
                )
                return TurnOutcome(
                    text=hit["answer"], thread_id=thread_id,
                    recent_turns=0, recalled_turns=0,
                    answer_cache_hit=True,
                )

        # Extraction BEFORE context assembly: if the user corrects a fact in
        # this very turn, the answer already reflects the new value. The LLM
        # call itself is inside the flag, not just the write.
        facts_written = 0
        dropped: list = []
        if self.extract_turns:
            raw, _model = extract.extract_facts(self.chat, text)
            if raw:
                valid, dropped = extract.validate_facts(raw)
                if valid:
                    written = extract.write_validated(
                        conn, user_id=self.user_id, thread_id=thread_id,
                        facts=valid,
                    )
                    facts_written = sum(1 for w in written if w["changed"])

        persona_text = persona.format_persona(persona.read_active(conn, self.user_id))

        facts: dict = {}
        for kind, name in entity_refs or []:
            resolved = entity.resolve(conn, user_id=self.user_id,
                                      kind=kind, name=name)
            if resolved:
                facts[name] = {f"{kind}.{a}": v
                               for a, v in resolved["facts"].items()}
        # The applicant's own facts (first-person statements bound to SELF)
        # are always in scope for the current user.
        self_resolved = entity.resolve(conn, user_id=self.user_id,
                                       kind="applicant", name=self.user_id)
        if self_resolved and self_resolved["facts"]:
            facts["you"] = {f"applicant.{a}": v
                            for a, v in self_resolved["facts"].items()}
        facts_text = "\n".join(
            f"{'' if name == 'you' else name + ' '}{k} = {v}"
            for name, kv in facts.items() for k, v in kv.items()
        )

        # Working memory: fold turns that fell out of the verbatim window.
        # Current facts are injected so the summary never re-asserts a stale
        # conversation value as current.
        summary = working.summarize_older(
            conn, user_id=self.user_id, thread_id=thread_id,
            chat=self.chat, keep_k=self.recency_k,
            current_facts=facts_text or None,
        )

        recent = conversation.recent_turns(conn, thread_id, k=self.recency_k)
        recent_before_current = [t for t in recent if t["seq"] < seq]
        recalled = conversation.recall(
            conn, user_id=self.user_id, thread_id=thread_id,
            query_embedding=query_vec, k=self.recall_k,
            before_seq=seq - len(recent_before_current),
        )
        chunks = knowledge.hybrid_search(
            conn, query=text, query_embedding=query_vec, k=4,
            min_score=self.kb_min_score,
        )

        assembled = context.assemble(
            system_base=SYSTEM_BASE,
            persona_text=persona_text,
            facts_text=facts_text,
            summary=summary,
            recalled=recalled,
            chunks=chunks,
            recent_before_current=recent_before_current,
            max_context_tokens=self.max_context_tokens,
        )
        chunks = assembled.chunks  # budget may have trimmed the excerpts

        messages = [{"role": "system", "content": assembled.system}]
        messages += assembled.messages
        messages.append({"role": "user", "content": text})

        tool_outcomes = self._tool_loop(
            thread_id=thread_id, messages=messages, text=text,
            query_vec=query_vec, facts=facts,
        )

        out = self.chat.complete(messages)
        # Verification: the tool numbers are authoritative; if the draft
        # omitted them, one retry with an explicit note. Code checks, the
        # model answers — arithmetic never trusted to prose.
        miss = _missing_numbers(out.text, tool_outcomes, text)
        if miss:
            messages.append({
                "role": "user",
                "content": (
                    "## Verification\nYour previous draft did not state the "
                    f"tool-computed value(s) {miss}, which are authoritative. "
                    "Final answer, using those numbers." 
                ),
            })
            retry = self.chat.complete(messages)
            if not _missing_numbers(retry.text, tool_outcomes, text):
                out = retry

        # Supersession guard: a draft that states a superseded value and omits
        # the current one is presenting stale memory as current. Rewrite once
        # with the authoritative values. Skipped for explicit recall questions,
        # where the old value is the point ("what did we say earlier?").
        if not any(w in text.lower() for w in _RECALL_WORDS):
            pairs = entity.superseded_pairs(conn, user_id=self.user_id)
            stale = _stale_hits(out.text, pairs)
            if stale:
                note = "; ".join(
                    f"{p['attr']} was {p['old']} and is now {p['current']}"
                    for p in stale)
                messages.append({"role": "user", "content": (
                    "## Correction\nThe draft presents superseded value(s). "
                    f"Current authoritative value(s): {note}. Rewrite the "
                    "answer using the current values; do not present a "
                    "superseded value as current.")})
                retry = self.chat.complete(messages)
                if not _stale_hits(retry.text, pairs):
                    out = retry

        if cacheable:
            cache.store(
                conn, user_id=self.user_id, thread_id=thread_id,
                question=text, query_embedding=query_vec, answer=out.text,
                ttl_seconds=self.cache_ttl,
            )

        conversation.append_turn(
            conn, user_id=self.user_id, thread_id=thread_id,
            role="assistant", content=out.text,
        )
        return TurnOutcome(
            text=out.text,
            thread_id=thread_id,
            recent_turns=len(recent_before_current),
            recalled_turns=assembled.recalled_turns,
            facts_used=facts,
            facts_written=facts_written,
            dropped_facts=dropped,
            retrieved_refs=[
                {"chunk_id": c["chunk_id"], "title": c["title"],
                 "chunk_index": c["chunk_index"], "preview": c["text"][:240]}
                for c in chunks
            ],
            tool_calls=[
                {k: t[k] for k in ("name", "source", "result_bytes",
                                   "offloaded")}
                for t in tool_outcomes
            ],
            tool_results=[
                {"name": t["name"], "args": t["args"], "source": t["source"],
                 "result": t["result_full"]}
                for t in tool_outcomes
            ],
            tokens=out.total_tokens,
            prompt_tokens=out.prompt_tokens,
            context_tokens=assembled.context_tokens,
            summary_used=assembled.summary_used,
            summary_turns=(
                len(summary["source_turn_ids"]) if summary else 0
            ),
        )

    # ------------------------------------------------------------------
    # Tool loop (unit 5)
    # ------------------------------------------------------------------
    def _tool_loop(
        self, *, thread_id: str, messages: list[dict], text: str,
        query_vec, facts: dict,
    ) -> list[dict]:
        """Plan/run tool calls until the planner says answer or rounds run out."""
        return toolbox.run_tool_loop(
            self.conn, user_id=self.user_id, thread_id=thread_id,
            chat=self.chat, messages=messages, text=text, query_vec=query_vec,
            facts=facts, use_tools=self.use_tools, tool_rounds=self.tool_rounds,
        )
