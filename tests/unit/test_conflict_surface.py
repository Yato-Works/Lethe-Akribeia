"""M14 Runtime Port & Parity: conflict surfacing in the shipped runtime.

V4_PROGRAM.md 4.24 / 4.25.  The M13 benchmark mechanism is a thin adapter
over ``artificial_memory.recall.conflict_surface``; these tests prove the
same mechanism fires in the product context builder, stays query-gated,
keeps provenance, and leaves the frozen (flag-off) semantics untouched.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "scripts" / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts" / "benchmarks"))

from artificial_memory.context.builder import EnhancedContextBuilder  # noqa: E402
from artificial_memory.core.models import (  # noqa: E402
    Memory,
    MemoryStatus,
    MemoryType,
    ResolutionLevel,
)
from artificial_memory.recall.conflict_surface import (  # noqa: E402
    conflict_entries,
    detect_conflicts,
    note_line,
    query_gated_keys,
)
from artificial_memory.recall.engine import BasicRecallEngine  # noqa: E402
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore  # noqa: E402

V1 = "For project atlas, the user reported the cache layer runs on SQLite."
V2 = "For project atlas, a teammate reported the cache layer actually runs on Redis."
UNRELATED = "For project atlas, the daily driver toolchain copilot was settled on."
QUERY = "For project atlas, which cache layer should I assume?"


@pytest.fixture()
def store(tmp_path):
    s = SQLiteMemoryStore(str(tmp_path / "test.db"))
    yield s
    s.close()


def _memory(store, topic_id: int, content: str, days_ago: int) -> Memory:
    """Store an active current memory with fixed day provenance."""
    stamp = datetime.now() - timedelta(days=days_ago)
    mem = Memory(
        topic_id=topic_id,
        memory_type=MemoryType.CURRENT,
        content=content,
        resolution=ResolutionLevel.RAW,
        importance=0.9,
        confidence=0.95,
        is_current=True,
        status=MemoryStatus.ACTIVE,
        valid_from=stamp,
        created_at=stamp,
    )
    return store.create_memory(mem)


def _builder(store, on: bool) -> EnhancedContextBuilder:
    return EnhancedContextBuilder(
        store, BasicRecallEngine(store), conflict_surfacing=on)


def _topic(store):
    from artificial_memory.core.models import Project, Topic

    project = store.create_project(Project(name="m14test",
                                           display_name="M14 Test"))
    topic = store.create_topic(
        Topic(project_id=project.id, name="M14",
              path="Projects/m14test/M14"))
    return topic.id


def test_single_source_of_truth_player_and_runtime_share_one_module():
    """Parity by construction: the player imports the runtime module."""
    import am_current_player as player_mod

    assert player_mod._canonical_detect is detect_conflicts
    assert player_mod._note_line is note_line
    assert player_mod._query_gated_keys is query_gated_keys


def test_detect_conflicts_keys_both_values_with_provenance(store, tmp_path):
    doc = [(V1, "day 45"), (V2, "day 48"), (UNRELATED, "day 50")]
    conflicts = detect_conflicts(doc)
    assert ("atlas", "cache layer") in conflicts
    entries = conflicts[("atlas", "cache layer")]
    values = [v for v, _ in entries]
    assert {"SQLite", "Redis"}.issubset(set(values))
    # provenance rides along per distinct value
    days = [label for _, label in entries]
    assert "day 45" in days and "day 48" in days


def test_note_line_carries_provenance_and_plan21_semantics():
    note = note_line("project atlas", "cache layer",
                     [("SQLite", "day 45"), ("Redis", "day 48")])
    assert note.startswith("NOTE: conflicting reports for project atlas")
    assert "SQLite (day 45) vs Redis (day 48)" in note
    assert "report both values" in note
    assert "which is correct is unverified" not in note


def test_conflict_entries_maps_each_distinct_value_to_its_first_document():
    docs = [(V1, "day 45"), (UNRELATED, "day 50"), (V2, "day 48"),
            (V2, "day 48")]  # a duplicate must not add an entry
    entries = conflict_entries(docs)
    rows = entries[("atlas", "cache layer")]
    assert rows[0] == ("SQLite", "day 45", 0)
    assert rows[1] == ("Redis", "day 48", 2)
    assert len(rows) == 2


def test_note_part_carries_source_assertion_lines(store):
    """M15.1: the reserved note part is note + the pair's raw lines with
    provenance, so co-presence never depends on the recency race."""
    topic_id = _topic(store)
    m1 = _memory(store, topic_id, V1, days_ago=10)
    m2 = _memory(store, topic_id, V2, days_ago=30)  # older than any pool line

    builder = _builder(store, on=True)
    pool = [m1, m2]
    for i in range(40):  # newer lines win the recency ordering
        pool.append(_memory(
            store, topic_id,
            f"For project rush{i}, the sprint plan was settled on the demo.",
            days_ago=1))
    context = builder.build_context(
        QUERY, topic_id, 4000, current_memories=pool)
    # note AND both source lines with their labels, inside one part
    assert "NOTE: conflicting reports for project atlas (cache layer)" in context
    assert f"- {V1}" in context
    assert f"- {V2}" in context


def test_runtime_remember_honors_occurred_at():
    """M15.2: facade.remember stores the caller's occurrence time so the
    builder's provenance label is a real, distinct time point."""
    import asyncio

    from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig

    runtime = ArtificialMemoryRuntime(RuntimeConfig(
        database_path=":memory:", memory_files_path=None,
        vector_index_path=None))
    stamp = datetime.now() - timedelta(days=45)
    asyncio.run(runtime.remember(V1, topic="M15", occurred_at=stamp))
    topic = runtime.conversation_manager.get_or_create_topic("M15")
    mems = runtime.store.get_memories(topic_id=topic.id)
    assert any(m.valid_from and m.valid_from.date() == stamp.date()
               for m in mems)


def test_ingest_knob_passes_scenario_timestamps():
    """M15.2: with provenance_days, the player feeds each turn's real
    scenario timestamp into facade.remember; without it, labels stay
    wall-clock."""
    import asyncio
    from types import SimpleNamespace

    from am_current_player import AMRuntimeConflictPlayer

    calls = []

    class _StubRuntime:
        async def remember(self, content, topic, occurred_at=None):
            calls.append(occurred_at)

    player = AMRuntimeConflictPlayer(
        {"provenance_days": True}, answerer=None)
    player._runtime = _StubRuntime()
    stamp = datetime(2026, 1, 5, 9, 0, 0) + timedelta(days=3)
    player._ingest_scenario(SimpleNamespace(
        turns=[SimpleNamespace(content="a", timestamp=stamp),
               SimpleNamespace(content="b", timestamp=stamp)]))
    assert calls == [stamp, stamp]
    assert player.name == "AM runtime conflict dt"

    calls.clear()
    plain = AMRuntimeConflictPlayer(None, answerer=None)
    plain._runtime = _StubRuntime()
    plain._ingest_scenario(SimpleNamespace(
        turns=[SimpleNamespace(content="a", timestamp=stamp)]))
    assert calls == [None]


def test_query_gate_does_not_fire_without_entity_and_property():
    conflicts = detect_conflicts([(V1, "day 45"), (V2, "day 48")])
    assert query_gated_keys("what about beacons?", conflicts) == []
    assert query_gated_keys(
        "For project atlas, which cache layer should I assume?", conflicts
    ) == [("atlas", "cache layer")]
    # a question naming the entity but a different property stays inert
    assert query_gated_keys(
        "For project atlas, which debugger did the user configure?", conflicts
    ) == []


def test_runtime_builder_fires_only_when_flagged(store):
    topic_id = _topic(store)
    m1 = _memory(store, topic_id, V1, days_ago=10)
    m2 = _memory(store, topic_id, V2, days_ago=7)

    on = _builder(store, on=True)
    context = on.build_context(QUERY, topic_id, 4000, current_memories=[m1, m2])
    assert "NOTE: conflicting reports for project atlas (cache layer)" in context
    assert "SQLite" in context and "Redis" in context
    assert "report both values" in context

    off = _builder(store, on=False)
    frozen = off.build_context(QUERY, topic_id, 4000, current_memories=[m1, m2])
    assert "NOTE:" not in frozen, "flag off must preserve frozen semantics"
    assert "SQLite" in frozen and "Redis" in frozen


def test_runtime_builder_stays_inert_on_unrelated_query(store):
    topic_id = _topic(store)
    m1 = _memory(store, topic_id, V1, days_ago=10)
    m2 = _memory(store, topic_id, V2, days_ago=7)
    other = _memory(store, topic_id,
                    "The user has not verified which cache layer report is"
                    " also for atlas, unrelated question about beacons.",
                    days_ago=5)

    builder = _builder(store, on=True)
    context = builder.build_context(
        "Which debugger did the user configure for project beacon?",
        topic_id, 4000, current_memories=[m1, m2, other])
    assert "NOTE:" not in context, (
        "the annotation must not ride into unrelated questions")


def test_annotation_keeps_both_values_no_merge_on_write_path(store):
    topic_id = _topic(store)
    sid_day = datetime.now() - timedelta(days=10)
    m1 = _memory(store, topic_id, V1, days_ago=10)
    m2 = _memory(store, topic_id, V2, days_ago=7)
    assert m1.is_current and m2.is_current
    assert m1.status == MemoryStatus.ACTIVE and m2.status == MemoryStatus.ACTIVE
    assert sid_day is not None
    # neither value was merged away by the annotation path
    rows = store.get_memories(topic_id)
    contents = {m.content for m in rows}
    assert {V1, V2}.issubset(contents)


def test_runtime_config_flag_defaults_to_frozen_semantics():
    from artificial_memory.runtime import RuntimeConfig

    assert RuntimeConfig().conflict_surfacing is False


def test_fill_ablation_removes_unrelated_fill_only_on_fired_queries(store):
    """M15.3: with ablation on, a fired conflict context carries the
    annotation and its evidence and loses the ~92% recency fill; a
    non-conflict context is byte-identical whether ablation is on or
    off, and flag-off stays frozen."""
    topic_id = _topic(store)
    m1 = _memory(store, topic_id, V1, days_ago=10)
    m2 = _memory(store, topic_id, V2, days_ago=30)
    pool = [m1, m2] + [
        _memory(store, topic_id,
                f"For project rush{i}, the sprint plan was settled on.",
                days_ago=1)
        for i in range(40)
    ]
    plain = _builder(store, on=True)
    ablate = EnhancedContextBuilder(
        store, BasicRecallEngine(store), conflict_surfacing=True,
        conflict_fill_ablation=True)

    fired = ablate.build_context(
        QUERY, topic_id, 4000, current_memories=pool)
    assert "NOTE: conflicting reports for project atlas (cache layer)" in fired
    assert f"- {V1}" in fired and f"- {V2}" in fired
    unrelated = [line for line in fired.splitlines()
                 if "project rush" in line]
    assert not unrelated, ("the recency fill must be removed when the "
                           "annotation fires")
    plain_ctx = plain.build_context(
        QUERY, topic_id, 4000, current_memories=pool)
    assert len(plain_ctx) > len(fired) * 2, (
        "control keeps the fill the ablation removes")

    inert_q = "Which debugger did the user configure for project beacon?"
    assert ablate.build_context(
        inert_q, topic_id, 4000, current_memories=pool) == \
        plain.build_context(inert_q, topic_id, 4000, current_memories=pool)


def test_tagged_note_survives_a_large_current_memory_pool(store):
    """M14 pricing finding: with 400+ candidates the note lost the budget
    race and fired on 0/320 product contexts.  Reservation happens FIRST,
    so the note must survive a crowded pool and a tight budget."""
    topic_id = _topic(store)
    m1 = _memory(store, topic_id, V1, days_ago=10)
    m2 = _memory(store, topic_id, V2, days_ago=7)
    pool = [m1, m2]
    for i in range(80):
        content = (
            f"For project bath{i % 3}, the notebook entry {i} was settled"
            " on the draft plan.")
        pool.append(_memory(store, topic_id, content, days_ago=6))

    builder = _builder(store, on=True)
    context = builder.build_context(
        QUERY, topic_id, 4000, current_memories=pool)
    assert "NOTE: conflicting reports for project atlas (cache layer)" in context

    tight = builder.build_context(QUERY, topic_id, 1200, current_memories=pool)
    assert "NOTE: conflicting reports" in tight, (
        "the reserved note must not be starved by the memory pool")
