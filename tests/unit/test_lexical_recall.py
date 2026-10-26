"""The lexical recall channel must prefer a rare shared term.

Measured motivation (controlled arena, 200 planted-fact questions): the runtime
answered 13.5%.  Its candidate pool was query-independent (40 episodes + 20
semantic out of 440 facts) and its ranking was unweighted word overlap, so the
asked fact was usually not even a candidate.  With the IDF-weighted channel the
gold reaches the context for 75% of questions, and on a paired 50-question arena
subset accuracy went 10.0% -> 54.0%.

These tests pin the two properties that make that work, on a synthetic store:

1. The channel is OFF by default (byte-identical baseline).
2. When on, a memory sharing a RARE query term outranks memories that share only
   common words, even though the common words are more numerous.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.core.models import (  # noqa: E402
    Conversation,
    Memory,
    MemoryStatus,
    MemoryType,
    Project,
    ResolutionLevel,
    Topic,
)
from artificial_memory.recall.engine import BasicRecallEngine  # noqa: E402
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore  # noqa: E402

COMMON = "the user said the project is fine"
FACT_ATLAS = "for project atlas the user prefers ripgrep as the editor toolchain"
FACT_TUNDRA = "for project tundra the user prefers sccache as the editor toolchain"
QUERY = "for project atlas what editor toolchain does the user prefer"


def _store_with(rows: list[str]):
    store = SQLiteMemoryStore(":memory:")
    project = store.create_project(Project(name="t"))
    topic = store.create_topic(Topic(project_id=project.id, name="Arena", path="Arena"))
    conv = store.create_conversation(Conversation(topic_id=topic.id, title="c")).id
    for content in rows:
        store.create_memory(Memory(
            topic_id=topic.id,
            memory_type=MemoryType.SEMANTIC,
            content=content,
            source_conversation_id=conv,
            resolution=ResolutionLevel.EPISODE,
            status=MemoryStatus.ACTIVE,
        ))
    return store, topic.id


def _top_contents(engine, topic_id, limit=3):
    memories, _ = engine.recall(QUERY, topic_id, level=2, max_tokens=4000)
    return [m.content for m in memories[:limit]]


class TestLexicalChannelIsOffByDefault:
    def test_default_engine_has_the_channel_disabled(self):
        store, _ = _store_with([COMMON, FACT_ATLAS])
        assert BasicRecallEngine(store).lexical is False

    def test_enabled_engine_reports_it(self):
        store, _ = _store_with([COMMON, FACT_ATLAS])
        assert BasicRecallEngine(store, lexical=True).lexical is True


class TestRareTermWins:
    def test_atlas_fact_reaches_the_context_with_the_channel(self):
        rows = [COMMON] * 6 + [FACT_TUNDRA, FACT_ATLAS]
        store, topic_id = _store_with(rows)
        engine = BasicRecallEngine(store, lexical=True)
        top = _top_contents(engine, topic_id, limit=2)
        assert any("atlas" in c for c in top), top

    def test_the_shared_platform_fact_does_not_win(self):
        rows = [COMMON] * 6 + [FACT_TUNDRA, FACT_ATLAS]
        store, topic_id = _store_with(rows)
        engine = BasicRecallEngine(store, lexical=True)
        top = _top_contents(engine, topic_id, limit=1)
        assert "tundra" not in top[0], top

    def test_the_channel_does_not_change_the_off_path(self):
        """Two engines over the same store: the off path must be identical."""
        rows = [COMMON] * 6 + [FACT_TUNDRA, FACT_ATLAS]
        store, topic_id = _store_with(rows)
        off_a = _top_contents(BasicRecallEngine(store), topic_id, limit=3)
        off_b = _top_contents(BasicRecallEngine(store), topic_id, limit=3)
        assert off_a == off_b
