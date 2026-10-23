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

import json
from dataclasses import dataclass, field

from agentmem.providers.chat import ChatClient
from agentmem.providers.embed import Embedder
from agentmem.tools import TOOLS
from agentmem.units import conversation, entity, extract, knowledge, persona, toolbox

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
        )

    # ------------------------------------------------------------------
    # Tool loop (unit 5)
    # ------------------------------------------------------------------
    def _tool_loop(
        self, *, thread_id: str, messages: list[dict], text: str,
        query_vec, facts: dict,
    ) -> list[dict]:
        """Plan/run tool calls until the planner says answer or rounds run out.

        Rules: the planner computes nothing itself; tool results arrive as
        user-role blocks labeled fresh/cache; each result is logged to
        mem.tool_call with its args hash so reuse is real, not claimed.
        """
        outcomes: list[dict] = []
        seen_hashes: set[str] = set()
        menu = ""
        if self.use_tools:
            found = toolbox.search_tools(self.conn, query_vec, k=4)
            menu = "\n".join(
                f"[{t['name']}] {t['description']} "
                f"params: {json.dumps(t['params'])}"
                for t in found
            )
            # Ranked tools first; the full registry (with exact param names)
            # always follows so the planner can never plan argument-blind.
            menu += "\n" + "\n".join(
                f"[{n}] params: {json.dumps(s['params'])} ttl={s['ttl']}s"
                for n, s in TOOLS.items()
            )
        rounds = 0
        while rounds < self.tool_rounds:
            plan_out = self.chat.complete(
                [
                    {"role": "system",
                     "content": toolbox.PLANNER_PROMPT.format(menu=menu)},
                    {"role": "user",
                     "content": toolbox.planner_payload(
                         question=text,
                         facts=facts,
                         results_so_far=[
                             {"tool": t["name"],
                              "result": json.loads(t["result_for_model"])
                              if t["result_for_model"].startswith("{")
                              else t["result_for_model"]}
                             for t in outcomes
                         ],
                     )},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            try:
                decision = json.loads(plan_out.text)
            except json.JSONDecodeError:
                break
            name = decision.get("name")
            if decision.get("action") != "tool" or name not in TOOLS:
                break
            args = decision.get("arguments") or {}
            h = toolbox.args_hash(name, args)
            if h in seen_hashes:
                # The planner asked for an identical call this turn; the
                # result is already in context. Re-computing teaches the
                # planner nothing and doubles cost for nothing.
                messages.append({
                    "role": "user",
                    "content": (f"## Duplicate request\n{name} was already "
                                "called with exactly these arguments this "
                                "turn; its result is above. Respond with "
                                '{"action": "answer"} now.'),
                })
                rounds += 1
                continue
            seen_hashes.add(h)
            try:
                t = toolbox.run_tool(
                    self.conn, user_id=self.user_id, thread_id=thread_id,
                    name=name, args=args,
                )
                outcomes.append(t)
                note = (f"## Tool result — {t['name']} ({t['source']})\n"
                        f"{t['result_for_model']}\n\n"
                        "Use this result. Call another tool or give the "
                        "final answer. If any policy rule, fee, or limit is "
                        "involved, cite it as [title §chunk] from the "
                        "retrieved excerpts.")
            except (ValueError, KeyError, TypeError) as exc:
                note = (f"## Tool error — {decision.get('name')}\n{exc}\n"
                        "Fix the arguments or answer without the tool.")
            messages.append({"role": "user", "content": note})
            rounds += 1
        return outcomes
