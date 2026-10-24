"""Memory backends: the same agent surface built two ways.

- scratch:   hand-rolled SQL over mem.* tables (units 0-6).
- langgraph: LangGraph PostgresSaver (episodic) + PostgresStore (long-term).

Both expose `answer(*, thread_id, text, entity_refs) -> TurnOutcome`, so one
workload runs through both and the difference is measured, not claimed.
"""

from agentmem.backends.base import MemoryBackend
from agentmem.backends.scratch import ScratchBackend

__all__ = ["MemoryBackend", "ScratchBackend"]
