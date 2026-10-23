"""Agent core: the stateless baseline and the memory-wired agent.

MemoryAgent's context assembly, in order:
  1. persona (one active version per user, injected once)
  2. entity facts (current values only — superseded rows never leak)
  3. recency (the last k turns, as proper chat messages)
  4. semantic recall of older turns (labelled, so provenance is visible)

Every answer persists both turns; the user turn is embedded for future
recall. The assistant turn is not embedded (halves embedding calls; recall
targets what the user asked, not what we replied).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.units import conversation, entity, extract, knowledge, persona

SYSTEM_BASE = (
    "You are the loan-officer copilot for Acme Home Loans. Answer from the "
    "context you are given. If a fact about an applicant is not in your "
    "context, say you don't know it. Quote policy numbers only when they "
    "appear in retrieved knowledge; never invent rates."
)


@dataclass
class TurnOutcome:
    text: str
    thread_id: str
    recent_turns: int = 0
    recalled_turns: int = 0
    facts_used: dict = field(default_factory=dict)
    facts_written: int = 0
    dropped_facts: list = field(default_factory=list)
    retrieved_chunks: int = 0
    tokens: int = 0


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
    ):
        self.conn = conn
        self.user_id = user_id
        self.chat = chat
        self.embedder = embedder
        self.recency_k = recency_k
        self.recall_k = recall_k
        self.kb_min_score = kb_min_score
        self.extract_turns = extract_turns
        persona.ensure_default(conn, user_id)

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

        # Extraction BEFORE context assembly: if the user corrects a fact in
        # this very turn, the answer already reflects the new value.
        facts_written = 0
        dropped: list = []
        raw, _model = extract.extract_facts(self.chat, text)
        if self.extract_turns and raw:
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
        facts_text = "\n".join(f"{name}: {kv}" for name, kv in facts.items())

        recent = conversation.recent_turns(conn, thread_id, k=self.recency_k)
        recent_before_current = [t for t in recent if t["seq"] < seq]
        recalled = conversation.recall(
            conn, user_id=self.user_id, thread_id=thread_id,
            query_embedding=query_vec, k=self.recall_k,
            before_seq=seq - len(recent_before_current),
        )

        blocks = [SYSTEM_BASE]
        if persona_text:
            blocks.append(f"## Persona\n{persona_text}")
        if facts_text:
            blocks.append(f"## Known facts\n{facts_text}")
        if recalled:
            blocks.append("## Earlier in this thread (semantically relevant)\n"
                          + conversation.format_turns(recalled))
        chunks = knowledge.hybrid_search(
            conn, query=text, query_embedding=query_vec, k=4,
            min_score=self.kb_min_score,
        )
        if chunks:
            blocks.append(
                "## Policy excerpts (cite as [title §chunk]; quote numbers "
                "only from here)\n" + knowledge.format_chunks(chunks)
            )
        system_prompt = "\n\n".join(blocks)

        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": t["role"], "content": t["content"]}
                     for t in recent_before_current]
        messages.append({"role": "user", "content": text})

        out = self.chat.complete(messages)
        conversation.append_turn(
            conn, user_id=self.user_id, thread_id=thread_id,
            role="assistant", content=out.text,
        )
        return TurnOutcome(
            text=out.text,
            thread_id=thread_id,
            recent_turns=len(recent_before_current),
            recalled_turns=len(recalled),
            facts_used=facts,
            facts_written=facts_written,
            dropped_facts=dropped,
            retrieved_chunks=len(chunks),
            tokens=out.total_tokens,
        )
