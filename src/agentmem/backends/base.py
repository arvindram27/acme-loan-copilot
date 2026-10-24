"""The backend seam: a structural contract, not a runtime constraint.

A MemoryBackend is anything that answers one user turn given the app-supplied
scope (user_id lives on the instance; thread_id is a per-call argument). The
shared TurnOutcome is the graded trace, so the evaluation harness treats both
backends identically.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentmem.agent import TurnOutcome


@runtime_checkable
class MemoryBackend(Protocol):
    user_id: str

    def answer(
        self,
        *,
        thread_id: str,
        text: str,
        entity_refs: list[tuple[str, str]] | None = None,
    ) -> TurnOutcome:
        """One user turn in, one assistant turn out; memory read and written."""
        ...
