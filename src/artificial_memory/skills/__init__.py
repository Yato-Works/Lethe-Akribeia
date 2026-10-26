"""AM Memory Skills - Deterministic Co-processor Framework.

The runtime's deterministic co-processor stack. The master entry point is
:class:`~artificial_memory.skills.deterministic_memory_engine.DeterministicMemoryEngine`,
which dispatches question-shaped messages to autonomous subsystems (counting,
temporal algebra, relational traversal, events, boolean verification, entity
attributes) and returns a committed answer when one engine is confident.
"""

from __future__ import annotations

from .answer_committer import CommittedAnswer, parse_turns
from .deterministic_memory_engine import DeterministicMemoryEngine
from .evidence_commit_gate import EvidenceCommitGate

__all__ = [
    "DeterministicMemoryEngine",
    "CommittedAnswer",
    "parse_turns",
    "EvidenceCommitGate",
]
