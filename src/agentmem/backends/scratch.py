"""Scratch backend: the hand-rolled SQL memory stack (units 0-6).

MemoryAgent wires conversation, persona, entity memory, KB, toolbox, working
memory and the semantic cache directly against mem.* tables with the scope
rule applied by hand. It IS the scratch backend; the class here exists only
to give it a stable name alongside the LangGraph backend.
"""

from __future__ import annotations

from agentmem.agent import MemoryAgent


class ScratchBackend(MemoryAgent):
    """Hand-rolled SQL over mem.* — see agent.MemoryAgent."""
