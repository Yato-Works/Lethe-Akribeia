"""Unit tests for PPREvidenceGraph (HippoRAG-style associative retrieval)."""

from __future__ import annotations

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.recall.ppr_graph import PPREvidenceGraph


def test_ppr_graph_construction_and_extraction() -> None:
    graph = PPREvidenceGraph()
    text = "Caroline moved to New York in April 2022 to work as an Engineer."
    entities = graph.extract_entities(text)

    assert "caroline" in entities
    assert "new york" in entities or "york" in entities
    assert "engineer" in entities
    assert "and" not in entities


def test_ppr_multi_hop_cross_session_association() -> None:
    """Test that PPR surfaces a relevant turn from another session via an intermediate entity link."""
    graph = PPREvidenceGraph()

    # Session 1: Mentions Project Atlas and leader Dave
    graph.add_turn(
        "t1",
        "[s1 on 2023-01-10] user: Dave is leading the development of Project Atlas.",
        session_id="s1",
    )
    # Session 2: Mentions Project Atlas uses ClickHouse for analytics
    graph.add_turn(
        "t2",
        "[s2 on 2023-02-15] user: For Project Atlas, we picked ClickHouse as our primary storage.",
        session_id="s2",
    )
    # Session 3: Distractor turn
    graph.add_turn(
        "t3",
        "[s3 on 2023-03-01] user: We ordered pizza and played board games all afternoon.",
        session_id="s3",
    )

    # Query connects to Dave, and should surface ClickHouse (t2) via Project Atlas link
    ranked = graph.rank_turns("What database did Dave's project choose?", top_k=2)
    assert len(ranked) >= 2

    top_ids = [t[0] for t in ranked]
    # Both t1 and t2 should be ranked above the distractor t3
    assert "t1" in top_ids
    assert "t2" in top_ids
    assert "t3" not in top_ids or ranked[0][1] > ranked[1][1]


def test_ppr_graph_build_from_records() -> None:
    records = [
        StructuredIR(
            entity="Melanie",
            property="art",
            value="painting sunrise",
            raw_content="[s1 on 2023-05-08] Melanie: I painted that lake sunrise last year!",
        ),
        StructuredIR(
            entity="Melanie",
            property="running",
            value="charity race",
            raw_content="[s2 on 2023-05-25] Melanie: I ran a charity race last Saturday.",
        ),
    ]
    graph = PPREvidenceGraph()
    graph.build_from_records(records)

    ranked = graph.rank_turns("Tell me about Melanie's sunrise painting.", top_k=1)
    assert len(ranked) == 1
    assert "sunrise" in ranked[0][2].lower()


def test_msc_compiler_with_ppr_retrieval() -> None:
    """Verify that MinimumSufficientContextCompiler integrates PPR widening properly."""
    from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler

    compiler = MinimumSufficientContextCompiler(ppr_retrieval=True)
    records = [
        StructuredIR(
            entity="Sarah",
            property="hobby",
            value="birdwatching",
            raw_content="[s1 on 2023-04-10] Sarah: I went birdwatching in the wetlands.",
        ),
        StructuredIR(
            entity="Sarah",
            property="camera",
            value="Sony Alpha",
            raw_content="[s2 on 2023-04-12] Sarah: I bought a Sony Alpha camera for my wetlands trip.",
        ),
        StructuredIR(
            entity="Bob",
            property="lunch",
            value="ramen",
            raw_content="[s3 on 2023-04-15] Bob: I had ramen for lunch.",
        ),
    ]

    pcc = compiler.compile("What camera did Sarah use for birdwatching?", records)
    assert not pcc.is_abstention
    assert "Sony Alpha" in pcc.context_text
    assert "birdwatching" in pcc.context_text

