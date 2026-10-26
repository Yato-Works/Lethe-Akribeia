"""Cross-resolution deduplication in the context builder.

The four recall levels return overlapping candidates. Before dedup, the same
memory id could be appended once per level and paid for out of the same token
budget. On a real two-turn topic the measured overlap was 20 rows -> 15 repeated
ids and 1 repeated content body. These tests pin the exact-id and
whitespace-normalized-content rules.
"""

from __future__ import annotations

from artificial_memory.context.builder import EnhancedContextBuilder
from artificial_memory.core.models import Memory, MemoryType


class _RecallEngine:
    def __init__(self, by_level):
        self.by_level = by_level

    def recall(self, query, topic_id, level, max_tokens=4000):
        return self.by_level.get(level, []), 0


def _memory(memory_id: int, content: str) -> Memory:
    return Memory(
        id=memory_id,
        topic_id=1,
        memory_type=MemoryType.EPISODE,
        content=content,
        importance=0.8,
        confidence=0.9,
    )


def test_same_memory_id_is_kept_once_across_levels() -> None:
    from artificial_memory.core.models import RecallLevel

    m1 = _memory(1, "The release is September 30.")
    m2 = _memory(2, "The beta needs two weeks.")
    engine = _RecallEngine(
        {
            RecallLevel.CURRENT_ONLY: [m1],
            RecallLevel.LONG_TERM_SUMMARY: [m1, m2],
            RecallLevel.EPISODE: [m2],
            RecallLevel.LIGHT_COMPRESSION: [m1, m2],
        }
    )
    builder = EnhancedContextBuilder(store=object(), recall_engine=engine)  # type: ignore[arg-type]

    parts, _, _ = builder._build_context_parts("release", 1, 8000, None)

    assert [part.memory_id for part in parts] == [1, 2]


def test_identical_content_from_different_ids_is_kept_once() -> None:
    from artificial_memory.core.models import RecallLevel

    raw = _memory(10, "The release is September 30.")
    # Same text, different record: a duplicate for context purposes.
    semantic = _memory(11, "The   release is September 30.  ")
    other = _memory(12, "The beta needs two weeks.")
    engine = _RecallEngine(
        {
            RecallLevel.CURRENT_ONLY: [],
            RecallLevel.LONG_TERM_SUMMARY: [semantic],
            RecallLevel.EPISODE: [raw, other],
            RecallLevel.LIGHT_COMPRESSION: [semantic, other],
        }
    )
    builder = EnhancedContextBuilder(store=object(), recall_engine=engine)  # type: ignore[arg-type]

    parts, _, _ = builder._build_context_parts("release", 1, 8000, None)

    assert len(parts) == 2
    assert {" ".join(part.content.split()) for part in parts} == {
        "The release is September 30.",
        "The beta needs two weeks.",
    }
    assert sorted(part.memory_id for part in parts) == [11, 12]
