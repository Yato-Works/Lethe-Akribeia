import pytest
from artificial_memory.compiler.coverage.concept_expander import AbstractConceptExpander
from artificial_memory.compiler.coverage.self_anchor_expander import SelfAnchorExpander
from artificial_memory.compiler.coverage.coverage_planner import EvidenceCoveragePlanner
from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine


def test_concept_expander_clothing():
    q = "How many items of clothing do I need to pick up or return from a store?"
    concepts = AbstractConceptExpander.detect_concepts(q)
    assert "clothing" in concepts
    subqueries = AbstractConceptExpander.generate_subqueries(q)
    assert len(subqueries) > 0
    joined = " ".join(subqueries).lower()
    assert "jacket" in joined or "shirt" in joined or "shoes" in joined


def test_concept_expander_furniture():
    q = "How many pieces of furniture did I buy, assemble, sell, or fix in the past few months?"
    concepts = AbstractConceptExpander.detect_concepts(q)
    assert "furniture" in concepts
    subqueries = AbstractConceptExpander.generate_subqueries(q)
    assert len(subqueries) > 0
    joined = " ".join(subqueries).lower()
    assert "chair" in joined or "table" in joined or "bookshelf" in joined


def test_self_anchor_when_born():
    q = "How old was I when Alex was born?"
    reqs = SelfAnchorExpander.analyze(q)
    assert len(reqs) == 1
    req = reqs[0]
    assert req.query_type == "age_difference"
    assert "Alex" in req.entity_a
    assert "born" in req.subquery_a
    assert "birthday" in req.subquery_b or "born" in req.subquery_b


def test_self_anchor_older_than_me():
    q = "How many years older is my grandma than me?"
    reqs = SelfAnchorExpander.analyze(q)
    assert len(reqs) == 1
    assert "grandma" in reqs[0].entity_a
    assert "my age" in reqs[0].subquery_b


def test_dual_entity_conjunction():
    q = "How many days did I spend in total traveling in Hawaii and in New York City?"
    reqs = SelfAnchorExpander.analyze(q)
    assert len(reqs) == 1
    assert "Hawaii" in reqs[0].entity_a
    assert "New York City" in reqs[0].entity_b


def test_coverage_planner():
    q = "What is the total amount I spent on luxury items in the past few months?"
    planned = EvidenceCoveragePlanner.plan(q)
    assert planned.intent == "SUM"
    assert len(planned.concept_subqueries) > 0
    assert planned.minimum_expected_sessions >= 2


class DummyRetriever:
    def rank_turns(self, query, turns, top_k=30, reference_date=None):
        scored = []
        for t in turns:
            score = 1.0 if any(w in t[3].lower() for w in query.lower().split()) else 0.1
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for _, t in scored[:top_k]]
        turn_scores = {(t[0], t[3]): s for s, t in scored[:top_k]}
        return top, turn_scores


def test_coverage_engine_convergent_retrieval():
    turns = [
        ("S1", "2023-01-01", "User", "I bought a winter jacket today at the mall."),
        ("S2", "2023-01-02", "User", "I returned those pants to the store."),
        ("S3", "2023-01-03", "User", "Picked up the dry-cleaned suit."),
        ("S4", "2023-01-04", "User", "Had lunch with a friend."),
    ]
    engine = MultiSessionCoverageEngine()
    retriever = DummyRetriever()
    q = "How many items of clothing do I need to pick up or return?"
    sorted_turns, scores, guaranteed_sids = engine.retrieve_convergent_evidence(
        q, turns, retriever, base_top_k=2
    )
    assert len(sorted_turns) >= 3
    found_sids = {t[0] for t in sorted_turns}
    assert "S1" in found_sids
    assert "S2" in found_sids
