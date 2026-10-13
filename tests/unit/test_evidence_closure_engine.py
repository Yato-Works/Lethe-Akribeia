"""Unit tests for Evidence Closure Engine (ECE) - Planner, Retriever, Closure Loop, Graph, and Compiler v2.

Verifies:
1. Typed QueryPlan extraction (Write LLM = 0).
2. Multi-View Retrieval with RRF fusion across Lexical, Predicate, Entity, Temporal, and Neighborhood views.
3. Iterative Evidence Closure Loop with requirement satisfaction evaluation.
4. Honest unresolved tracking when requirements cannot be fulfilled.
5. Verifiable Evidence Graph construction preserving verbatim text and provenance.
6. Safety boundary conditions: empty QueryPlan, empty dialogue, zero-division prevention.
7. Conflicting temporal assertions (STATE_EVOLUTION) detection.
8. EvidenceCompilerV2 dynamic marginal utility: ΔR(b | S) = R(S ∪ {b}) - R(S) prevents duplicate bonus.
9. Runtime Gold data isolation (no gold answers/IDs in any method signature or object).
10. Bit-exact deterministic reproducibility.
"""

import pytest
from artificial_memory.compiler.closure.types import (
    AnswerTopology,
    RequirementStatus,
    EntityTarget,
    PredicateRequirement,
    EvidenceRequirement,
    QueryPlan,
    ClosureCandidate,
    ClosureState,
)
from artificial_memory.compiler.closure.planner import DeterministicQueryPlanner
from artificial_memory.compiler.closure.retriever import MultiViewRetriever
from artificial_memory.compiler.closure.closure_engine import EvidenceClosureEngine
from artificial_memory.compiler.closure.compiler_v2 import EvidenceCompilerV2, CompiledContextV2


def test_query_plan_types():
    target = EntityTarget(canonical_name="caroline", aliases=("Caroline",), is_speaker=True)
    assert target.canonical_name == "caroline"
    assert target.is_speaker is True

    req = EvidenceRequirement(
        requirement_id="req_test",
        description="Test requirement",
        status=RequirementStatus.UNRESOLVED,
    )
    assert req.status == RequirementStatus.UNRESOLVED

    state = ClosureState()
    assert state.iteration == 0
    assert state.is_converged is False


def test_deterministic_planner_enumeration():
    planner = DeterministicQueryPlanner()
    plan = planner.plan("What LGBTQ+ events has Caroline participated in?")

    assert plan.topology == AnswerTopology.SET_ENUMERATION
    assert any(ent.canonical_name == "caroline" for ent in plan.entity_targets)
    assert any(pred.predicate_name == "participated in" for pred in plan.predicates)
    assert len(plan.requirements) >= 2
    assert any(r.requirement_id == "req_enum_instances" for r in plan.requirements)
    assert plan.recommended_budget >= 1200


def test_deterministic_planner_temporal():
    planner = DeterministicQueryPlanner()
    plan = planner.plan("When did Melanie paint a sunrise?")

    assert plan.topology == AnswerTopology.TEMPORAL_DURATION
    assert any(ent.canonical_name == "melanie" for ent in plan.entity_targets)
    assert any(pred.predicate_name == "paint" for pred in plan.predicates)


def test_deterministic_planner_inference():
    planner = DeterministicQueryPlanner()
    plan = planner.plan("Would Caroline likely have Dr. Seuss books?")

    assert plan.topology == AnswerTopology.INFERENCE_BOOLEAN
    assert any(ent.canonical_name == "caroline" for ent in plan.entity_targets)


def test_deterministic_planner_generic():
    planner = DeterministicQueryPlanner()
    # Test on completely unseen generic question outside LoCoMo
    plan = planner.plan("Where did the user travel last year?")

    assert plan.topology == AnswerTopology.SINGLE_FACT
    assert len(plan.temporal_constraints) == 1
    assert plan.temporal_constraints[0].time_expression == "last year"


def test_multiview_retriever_rrf():
    mock_turns = [
        ("session_1", "10 May, 2023", "Caroline", "I recently attended a local support group for trans youth."),
        ("session_1", "10 May, 2023", "Melanie", "That sounds really inspiring! Did they have any mentors?"),
        ("session_2", "15 June, 2023", "Caroline", "We went marching in the annual pride parade downtown."),
        ("session_2", "15 June, 2023", "Melanie", "I saw pictures of that, the flags were beautiful!"),
        ("session_3", "1 July, 2023", "Melanie", "I love cooking pasta on Sunday evenings."),
    ]

    planner = DeterministicQueryPlanner()
    plan = planner.plan("What events has Caroline participated in?")

    retriever = MultiViewRetriever(rrf_k=60)
    candidates = retriever.retrieve(plan, mock_turns, top_k=5)

    assert len(candidates) > 0
    top_turn_ids = [c.turn_id for c in candidates[:2]]
    assert "session_1:t0" in top_turn_ids or "session_2:t2" in top_turn_ids

    for c in candidates:
        assert c.rrf_score > 0.0
        assert len(c.view_ranks) > 0
        assert c.session_id in ("session_1", "session_2", "session_3")


def test_evidence_closure_loop_end_to_end():
    mock_turns = [
        ("session_1", "10 May, 2023", "Caroline", "I recently attended a local support group for trans youth."),
        ("session_1", "10 May, 2023", "Melanie", "That sounds really inspiring! How was it?"),
        ("session_2", "15 June, 2023", "Caroline", "We went marching in the annual pride parade downtown."),
        ("session_2", "15 June, 2023", "Melanie", "I loved the banners you made!"),
    ]

    planner = DeterministicQueryPlanner()
    plan = planner.plan("What activities has Caroline participated in?")

    engine = EvidenceClosureEngine()
    result = engine.run_closure(plan, mock_turns, initial_top_k=5)

    assert result.closure_completeness == 1.0
    assert result.closure_state.is_converged is True
    assert result.closure_state.termination_reason == "all_requirements_satisfied"
    assert len(result.unmet_requirements) == 0
    assert len(result.graph_nodes) == len(result.admitted_candidates)
    assert len(result.graph_edges) > 0

    edge_types = {e.relation_type for e in result.graph_edges}
    assert "DIALOGUE_FLOW" in edge_types
    assert "SAME_SPEAKER_REMENTION" in edge_types


def test_evidence_closure_unresolved_honesty():
    single_session_turns = [
        ("session_1", "10 May, 2023", "Caroline", "I made a clay vase today."),
        ("session_1", "10 May, 2023", "Melanie", "Nice work!"),
    ]

    planner = DeterministicQueryPlanner()
    plan = planner.plan("What activities has Caroline participated in across multiple sessions?")

    engine = EvidenceClosureEngine()
    result = engine.run_closure(plan, single_session_turns, initial_top_k=2)

    assert result.closure_completeness < 1.0
    assert "req_multi_session_breadth" in result.unmet_requirements
    assert result.closure_state.is_converged is False
    assert result.closure_state.termination_reason in ("expansion_exhausted", "no_further_expansion_possible", "max_hops_reached")


def test_closure_engine_empty_query_plan_safety():
    """Item 3: Empty QueryPlan must not crash with 0-division and must not report resolved."""
    empty_plan = QueryPlan(
        query_text="???",
        topology=AnswerTopology.SINGLE_FACT,
        entity_targets=(),
        predicates=(),
        temporal_constraints=(),
        requirements=(),
        recommended_budget=800,
        max_closure_hops=2,
    )
    engine = EvidenceClosureEngine()
    result = engine.run_closure(empty_plan, [("s1", "date", "speaker", "content")])

    assert result.closure_completeness == 0.0
    assert result.closure_state.termination_reason == "planning_incomplete"
    assert result.closure_state.is_converged is False


def test_closure_engine_empty_dialogue_safety():
    """Item 3: Empty dialogue input returns gracefully without unhandled exceptions."""
    planner = DeterministicQueryPlanner()
    plan = planner.plan("Where did Caroline travel?")
    engine = EvidenceClosureEngine()
    result = engine.run_closure(plan, [])

    assert result.closure_completeness == 0.0
    assert result.closure_state.termination_reason == "empty_dialogue"
    assert len(result.admitted_candidates) == 0


def test_closure_engine_conflicting_state_evolution():
    """Item 1: Temporal evolution markers flag CONFLICTING status."""
    conflicting_turns = [
        ("session_1", "10 May, 2022", "Caroline", "I used to love drinking dark roast coffee every morning."),
        ("session_4", "15 June, 2023", "Caroline", "I switched to green tea and now currently dislike coffee."),
    ]
    plan = QueryPlan(
        query_text="What is Caroline's coffee preference currently compared to previously?",
        topology=AnswerTopology.STATE_EVOLUTION,
        entity_targets=(EntityTarget(canonical_name="caroline", aliases=("Caroline",), is_speaker=True),),
        predicates=(PredicateRequirement(predicate_name="preference", category="state"),),
        temporal_constraints=(),
        requirements=(
            EvidenceRequirement(
                requirement_id="req_pref_state",
                description="Determine coffee preference and evolution",
                target_entity="caroline",
                target_predicate="preference",
            ),
        ),
        recommended_budget=1000,
    )
    engine = EvidenceClosureEngine()
    result = engine.run_closure(plan, conflicting_turns)

    req_map = {r.requirement_id: r for r in result.requirement_details}
    assert req_map["req_pref_state"].status == RequirementStatus.CONFLICTING
    assert "chronological evolution markers" in req_map["req_pref_state"].coverage_basis
    assert len(req_map["req_pref_state"].supporting_turn_ids) == 2


def test_evidence_compiler_v2_dynamic_gain():
    """Item 4: Dynamic marginal utility ΔR(b | S) prevents double-counting satisfied requirements."""
    # Two candidates both support req_enum_instances
    cand_a = ClosureCandidate(
        turn_id="D1:1",
        session_id="session_1",
        session_date="10 May, 2023",
        speaker="Caroline",
        content="I attended the LGBTQ youth conference yesterday.",
        rrf_score=0.05,
    )
    cand_b = ClosureCandidate(
        turn_id="D1:2",
        session_id="session_1",
        session_date="10 May, 2023",
        speaker="Caroline",
        content="The LGBTQ conference had incredible workshops.",
        rrf_score=0.04,
    )
    cand_c = ClosureCandidate(
        turn_id="D5:1",
        session_id="session_5",
        session_date="20 June, 2023",
        speaker="Caroline",
        content="Next month I will be running in the 5k charity marathon.",
        rrf_score=0.03,
    )

    plan = QueryPlan(
        query_text="What LGBTQ events did Caroline attend?",
        topology=AnswerTopology.SET_ENUMERATION,
        entity_targets=(EntityTarget(canonical_name="caroline"),),
        predicates=(PredicateRequirement(predicate_name="attend", category="action"),),
        temporal_constraints=(),
        requirements=(
            EvidenceRequirement(
                requirement_id="req_enum_instances",
                description="Events attended",
                status=RequirementStatus.SATISFIED,
                supporting_turn_ids=("D1:1", "D1:2"),
            ),
            EvidenceRequirement(
                requirement_id="req_other_charity",
                description="Other charity event",
                status=RequirementStatus.SATISFIED,
                supporting_turn_ids=("D5:1",),
            ),
        ),
        recommended_budget=500,
    )

    from artificial_memory.compiler.closure.types import EvidenceClosureResult, ClosureState
    closure_res = EvidenceClosureResult(
        query_plan=plan,
        closure_state=ClosureState(iteration=1, is_converged=True),
        admitted_candidates=(cand_a, cand_b, cand_c),
        unmet_requirements=(),
        graph_nodes=(),
        graph_edges=(),
        closure_completeness=1.0,
        requirement_details=plan.requirements,
    )

    compiler = EvidenceCompilerV2()
    compiled = compiler.compile(closure_res, token_budget=300)

    # Both distinct requirements must be resolved by prioritizing cand_a and cand_c,
    # rather than wasting budget on cand_b which offers zero incremental requirement gain ΔR(b | S)
    assert "D1:1" in compiled.admitted_turn_ids
    assert "D5:1" in compiled.admitted_turn_ids
    assert set(compiled.resolved_requirements) == {"req_enum_instances", "req_other_charity"}


def test_runtime_gold_data_isolation():
    """Item 5: Enforces zero runtime dependency or parameter accepting gold answer or gold evidence IDs."""
    planner = DeterministicQueryPlanner()
    retriever = MultiViewRetriever()
    engine = EvidenceClosureEngine()
    compiler = EvidenceCompilerV2()

    # Verify method signatures do NOT accept gold_answer, gold_id, or question_id
    import inspect
    for fn in [planner.plan, retriever.retrieve, engine.run_closure, compiler.compile]:
        sig = inspect.signature(fn)
        param_names = set(sig.parameters.keys())
        assert "gold_answer" not in param_names
        assert "gold_evidence" not in param_names
        assert "question_id" not in param_names
        assert "gold_ids" not in param_names


def test_deterministic_reproducibility():
    """Item 5: Running closure twice on identical input produces bit-exact identical candidates and scores."""
    mock_turns = [
        ("session_1", "10 May, 2023", "Caroline", "I attended a local community workshop."),
        ("session_2", "15 June, 2023", "Caroline", "I volunteered at the public library book sale."),
    ]
    planner = DeterministicQueryPlanner()
    plan = planner.plan("What activities has Caroline participated in?")
    engine = EvidenceClosureEngine()

    run1 = engine.run_closure(plan, mock_turns)
    run2 = engine.run_closure(plan, mock_turns)

    assert run1.closure_completeness == run2.closure_completeness
    assert run1.closure_state.termination_reason == run2.closure_state.termination_reason
    assert [c.turn_id for c in run1.admitted_candidates] == [c.turn_id for c in run2.admitted_candidates]
    assert [c.rrf_score for c in run1.admitted_candidates] == [c.rrf_score for c in run2.admitted_candidates]
