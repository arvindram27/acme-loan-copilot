"""LangGraph backend: the same memory semantics over LangGraph primitives.

Contrast with scratch (mem.* tables):
  conversation (episodic) -> langgraph PostgresSaver checkpointer
  persona                -> langgraph PostgresStore namespace (user_id, "persona")
  entity facts           -> langgraph PostgresStore namespace (user_id, "entity")
                            supersession is put() overwrite (current facts win);
                            cross-thread visibility is automatic (user-scoped,
                            not thread-scoped).
  semantic recall of old turns -> PostgresStore namespace (user_id, "turns",
                            thread_id) with an embedding index + search().

Everything that is application logic rather than memory stays shared: KB
retrieval (app-shared by design), fact extraction, the deterministic toolbox
and tool log, and context assembly. Working-memory summaries and the semantic
answer cache are scratch-only for now (documented gap; they re-implement
cleanly on the store but are out of this milestone's scope).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.postgres import PostgresStore

from agentmem.agent import (
    SYSTEM_BASE,
    TurnBudget,
    TurnBudgetExceeded,
    TurnOutcome,
    _BudgetedChat,
    _missing_numbers,
)
from agentmem.units import context, extract, knowledge, persona, toolbox


def build_store(conn, embedder, dims: int = 1536) -> PostgresStore:
    """PostgresStore with a semantic index over the "text" field of values.

    `conn` is a psycopg ConnectionPool or Connection dedicated to LangGraph —
    NOT shared with direct SQL (knowledge/toolbox), which gets its own
    connection. Sharing one raw connection corrupts psycopg's pipeline state.
    """
    store = PostgresStore(
        conn,
        index={"dims": dims, "embed": embedder.embed_batch, "fields": ["text"]},
    )
    store.setup()
    return store


def build_checkpointer(conn) -> PostgresSaver:
    """PostgresSaver for thread checkpoints (episodic memory).

    Same caveat as build_store: give it a dedicated ConnectionPool."""
    saver = PostgresSaver(conn)
    saver.setup()
    return saver


class LangGraphAgent:
    """A memory agent whose long-term + episodic memory live in LangGraph.

    user_id is the app-supplied scope; thread_id arrives per turn in the
    graph config. The store and checkpointer are shared across all users of a
    process; conn (the agentmem_lg connection) backs the shared KB and tool
    log.
    """

    def __init__(
        self,
        *,
        user_id: str,
        chat,
        embedder,
        conn,
        store: PostgresStore,
        checkpointer: PostgresSaver,
        recency_k: int = 10,
        recall_k: int = 4,
        kb_min_score: float = 0.30,
        extract_turns: bool = True,
        use_tools: bool = True,
        tool_rounds: int = 3,
        max_context_tokens: int = 2800,
        max_calls_per_turn: int = 8,
        turn_deadline_s: float = 90.0,
    ):
        self.user_id = user_id
        self.chat = chat
        self.embedder = embedder
        self.conn = conn
        self.store = store
        self.checkpointer = checkpointer
        self.recency_k = recency_k
        self.recall_k = recall_k
        self.kb_min_score = kb_min_score
        self.extract_turns = extract_turns
        self.use_tools = use_tools
        self.tool_rounds = tool_rounds
        self.max_calls_per_turn = max_calls_per_turn
        self.turn_deadline_s = turn_deadline_s
        self.max_context_tokens = max_context_tokens
        self._last_outcome: TurnOutcome | None = None
        self._ensure_persona()
        toolbox.sync_tools(conn, embedder)
        self.graph = self._build_graph()

    # -- graph ---------------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(MessagesState)
        g.add_node("agent", self._node)
        g.add_edge(START, "agent")
        g.add_edge("agent", END)
        return g.compile(checkpointer=self.checkpointer, store=self.store)

    def _node(self, state: dict, config: RunnableConfig) -> dict:
        thread_id = config["configurable"]["thread_id"]
        entity_refs = config["configurable"].get("entity_refs") or []
        messages = state["messages"]
        text = messages[-1].content
        seq = len(messages)  # 1-based count incl. the just-added user turn

        budget = TurnBudget(max_calls=self.max_calls_per_turn,
                            deadline_s=self.turn_deadline_s)
        chat = _BudgetedChat(self.chat, budget)

        # Extraction BEFORE context: a correction in this turn is reflected.
        # The LLM call sits inside the flag, so "off" really means off.
        facts_written = 0
        dropped: list = []
        if self.extract_turns:
            raw, _ = extract.extract_facts(chat, text)
            if raw:
                valid, dropped = extract.validate_facts(raw)
                if valid:
                    facts_written = self._write_facts(valid)

        persona_text = persona.format_persona(self._get_persona())

        facts: dict = {}
        for kind, name in entity_refs:
            resolved = self._resolve(kind, name)
            if resolved:
                facts[name] = {f"{kind}.{a}": v for a, v in resolved.items()}
        self_resolved = self._resolve("applicant", self.user_id)
        if self_resolved:
            facts["you"] = {f"applicant.{a}": v for a, v in self_resolved.items()}
        facts_text = "\n".join(
            f"{'' if name == 'you' else name + ' '}{k} = {v}"
            for name, kv in facts.items() for k, v in kv.items()
        )

        query_vec = self.embedder.embed(text)

        recent_before_current = self._recent(messages)
        recent_texts = {t["content"] for t in recent_before_current}
        recalled = [r for r in self._recall(thread_id, text)
                    if r["content"] not in recent_texts]
        # store AFTER recall so the current turn never recalls itself
        self._store_turn(thread_id, seq, text)

        chunks = knowledge.hybrid_search(
            self.conn, query=text, query_embedding=query_vec, k=4,
            min_score=self.kb_min_score,
        )
        assembled = context.assemble(
            system_base=SYSTEM_BASE,
            persona_text=persona_text,
            facts_text=facts_text,
            summary=None,  # working-memory summaries: scratch-only for now
            recalled=recalled,
            chunks=chunks,
            recent_before_current=recent_before_current,
            max_context_tokens=self.max_context_tokens,
        )
        chunks = assembled.chunks

        snapshot = {
            "facts_used": facts,
            "fact_ids": [],  # the LangGraph store has no row versions
            "summary_id": None,
            "summary_through": None,
            "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
            "cache_hit": False,
        }

        chat_messages = [{"role": "system", "content": assembled.system}]
        chat_messages += assembled.messages
        chat_messages.append({"role": "user", "content": text})

        tool_outcomes = toolbox.run_tool_loop(
            self.conn, user_id=self.user_id, thread_id=thread_id,
            chat=chat, messages=chat_messages, text=text,
            query_vec=query_vec, facts=facts,
            use_tools=self.use_tools, tool_rounds=self.tool_rounds,
        )

        out = chat.complete(chat_messages)
        miss = _missing_numbers(out.text, tool_outcomes, text)
        if miss:
            chat_messages.append({
                "role": "user",
                "content": (
                    "## Verification\nYour previous draft did not state the "
                    f"tool-computed value(s) {miss}, which are authoritative. "
                    "Final answer, using those numbers."
                ),
            })
            try:
                retry = chat.complete(chat_messages)
            except TurnBudgetExceeded:
                retry = None
            if retry is not None and not _missing_numbers(
                    retry.text, tool_outcomes, text):
                out = retry

        self._last_outcome = TurnOutcome(
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
                {k: t[k] for k in ("name", "source", "result_bytes", "offloaded")}
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
            summary_used=False,
            summary_turns=0,
            context_snapshot=snapshot,
        )
        return {"messages": [AIMessage(content=out.text)]}

    # -- memory reads/writes on the store ------------------------------------
    def _ensure_persona(self) -> None:
        if self.store.get((self.user_id, "persona"), "active") is None:
            self.store.put(
                (self.user_id, "persona"), "active",
                {"version": 1, "change_trigger": "app_default",
                 "body": persona.DEFAULT_PERSONA},
            )

    def _get_persona(self) -> dict | None:
        item = self.store.get((self.user_id, "persona"), "active")
        return item.value if item else None

    def _resolve(self, kind: str, name: str) -> dict | None:
        item = self.store.get((self.user_id, "entity"), f"{kind}:{name}")
        if item is None or not item.value:
            return None
        return item.value.get("facts") or None

    def _write_facts(self, valid: list[dict]) -> int:
        """Store facts with put()-overwrite supersession; returns changed count."""
        changed = 0
        for f in valid:
            name = self.user_id if f["name"] == "SELF" else f["name"]
            key = f"{f['kind']}:{name}"
            item = self.store.get((self.user_id, "entity"), key)
            facts = dict(item.value.get("facts") or {}) if item and item.value else {}
            if facts.get(f["attr"]) == f["value"]:
                continue
            facts[f["attr"]] = f["value"]
            self.store.put(
                (self.user_id, "entity"), key,
                {"kind": f["kind"], "name": name, "facts": facts},
            )
            changed += 1
        return changed

    def _store_turn(self, thread_id: str, seq: int, text: str) -> None:
        self.store.put(
            (self.user_id, "turns", thread_id), f"t{seq}",
            {"text": text, "role": "user", "seq": seq}, index=["text"],
        )

    def _recall(self, thread_id: str, text: str) -> list[dict]:
        try:
            results = self.store.search(
                (self.user_id, "turns", thread_id), query=text, limit=self.recall_k,
            )
        except Exception:
            return []
        out = []
        for r in results:
            v = r.value or {}
            out.append({
                "seq": v.get("seq", 0),
                "role": v.get("role", "user"),
                "content": v.get("text", ""),
                "similarity": float(getattr(r, "score", 0.0)),
            })
        return out

    def _recent(self, messages: list) -> list[dict]:
        """The verbatim window: message history before the current turn."""
        turns = []
        for i, m in enumerate(messages[:-1], start=1):
            role = "user" if m.type == "human" else "assistant"
            turns.append({"seq": i, "role": role, "content": m.content})
        return turns[-self.recency_k:] if len(turns) > self.recency_k else turns

    # -- public surface ------------------------------------------------------
    def answer(
        self,
        *,
        thread_id: str,
        text: str,
        entity_refs: list[tuple[str, str]] | None = None,
    ) -> TurnOutcome:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "entity_refs": entity_refs or [],
            }
        }
        self._last_outcome = None
        self.graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)
        assert self._last_outcome is not None, "graph node did not set outcome"
        return self._last_outcome
