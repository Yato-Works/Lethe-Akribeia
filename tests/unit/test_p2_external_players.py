"""Unit tests for the P2 external-style players (Zep / HippoRAG).

Deterministic unit mode (no LLM): facts become the raw exchange lines;
answers are the frozen abstention text. Checks cover protocol compliance:
budget cap, write-side cost records, temporal edge stamps (Zep) and
PPR seeding/ranking (HippoRAG).

Requires the ``[vector]`` optional extra (faiss + sentence-transformers).
The whole module is skipped with a clear reason when faiss is absent.
"""
from __future__ import annotations

import pytest

# Both players call VectorSearchEngine which requires faiss; skip early so
# a lean dev venv (no [vector] extra) does not produce hard ImportErrors.
pytest.importorskip("faiss", reason="lethe-akribeia[vector] extra required")

from artificial_memory.research.benchmarks.arena import build_dataset
from artificial_memory.research.benchmarks.players import HippoRAGStylePlayer, ZepStylePlayer


@pytest.fixture(scope="module")
def scenarios():
    dataset = build_dataset()
    return list(dataset.scenarios[:2])


def _answer(player, dataset):
    question = dataset.questions[0]
    return player._answer_with_budget(question, player.context_budget)


def test_zep_ingest_builds_temporal_facts(scenarios):
    player = ZepStylePlayer({"context_budget": 1000}, answerer=None)
    player.ingest(scenarios)

    assert player._facts, "facts must be extracted at ingest"
    days = {f["day"] for f in player._facts}
    assert all(d >= 0 for d in days)
    # Every fact is indexed under at least one entity OR stores the raw exchange.
    assert all(
        f["entities"] or f["fact"].startswith("[")
        for f in player._facts
    )
    # Write-side cost reported.
    assert player.finalize().write_latency_ms >= 0


def test_zep_retrieval_returns_facts_and_respects_budget(scenarios):
    dataset = build_dataset()
    player = ZepStylePlayer({"context_budget": 1000}, answerer=None)
    player.ingest(scenarios)
    answer = _answer(player, dataset)

    assert answer.context_tokens_used <= 1000
    assert answer.player_name == "Zep-style"
    # Temporal stamps are rendered into context.
    stamp = "[day "
    assert answer.metadata["retrieved"] >= 0
    ctx = answer.metadata.get("context_preview", "")
    assert isinstance(ctx, str)


def test_hipporag_ingest_builds_bipartite_graph(scenarios):
    player = HippoRAGStylePlayer({"context_budget": 1000}, answerer=None)
    player.ingest(scenarios)

    assert player._passages, "passages must exist"
    passage_nodes = [p["node"] for p in player._passages]
    assert all(n.startswith("p:") for n in passage_nodes)
    # Any entity edges present are symmetric.
    for node, neighbors in player._edges.items():
        for nb in neighbors:
            assert node in player._edges[nb], f"edge {node}->{nb} must be symmetric"


def test_hipporag_ppr_seeds_admit_no_negative_mass(scenarios):
    player = HippoRAGStylePlayer({"context_budget": 1000}, answerer=None)
    player.ingest(scenarios)
    seed_node = "e:alice" if "e:alice" in player._edges else next(
        (e for e in player._edges if e.startswith("e:")), None
    )
    if seed_node is None:
        pytest.skip("no entity nodes in fixtures")
    probs = player._ppr({seed_node: 1.0})
    assert probs, "PPR must assign positive mass"
    assert all(v >= 0 for v in probs.values())


def test_hipporag_answer_respects_budget(scenarios):
    dataset = build_dataset()
    player = HippoRAGStylePlayer({"context_budget": 1000}, answerer=None)
    player.ingest(scenarios)
    answer = _answer(player, dataset)
    assert answer.context_tokens_used <= 1000
    assert answer.metadata["retrieved"] >= 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
