"""Unit tests for EvidenceSufficiencyJudge and Adaptive Escalation."""

from __future__ import annotations

from artificial_memory.compiler.budgeted_compiler import (
    BudgetedCognitiveCompiler,
    CompiledContext,
)
from artificial_memory.compiler.graph.dependency_graph import (
    EvidenceBundle,
    EvidenceNode,
)
from artificial_memory.compiler.sufficiency_judge import (
    EvidenceSufficiencyJudge,
    SufficiencyAssessment,
)


def test_sufficiency_judge_evaluation_sufficient() -> None:
    judge = EvidenceSufficiencyJudge()

    ctx = CompiledContext(
        context_text="[S1] User: I visited Kyoto in May 2022.",
        total_tokens=700,
        selected_bundle_ids=["b1"],
        selected_node_ids=["n1"],
        utility_score=4.0,
        budget_limit=800,
        provenance_trail=[],
    )

    assessment = judge.evaluate(
        query="When did I visit Kyoto?",
        plan=None,
        compiled_context=ctx,
        top1_anchor_score=80.0,
        selected_session_count=1,
        is_multi_session=False,
        candidate_count=5,
    )

    assert assessment.is_sufficient
    assert assessment.sufficiency_score >= 0.50
    assert len(assessment.distress_reasons) == 0


def test_sufficiency_judge_evaluation_distress_budget_starvation() -> None:
    judge = EvidenceSufficiencyJudge()

    # Under-utilized budget (150 tokens on a 900 limit) and 0 utility
    ctx = CompiledContext(
        context_text="[S1] User: Hello there.",
        total_tokens=150,
        selected_bundle_ids=["b1"],
        selected_node_ids=["n1"],
        utility_score=0.0,
        budget_limit=900,
        provenance_trail=[],
    )

    assessment = judge.evaluate(
        query="What was the total expenditure on hotel bookings in Rome?",
        plan=None,
        compiled_context=ctx,
        top1_anchor_score=30.0,
        selected_session_count=1,
        is_multi_session=False,
        candidate_count=10,
    )

    assert not assessment.is_sufficient
    assert "budget_starvation" in assessment.distress_reasons
    assert "anchor_prefill" in assessment.recommended_escalations


def test_budgeted_compiler_adaptive_escalation() -> None:
    compiler = BudgetedCognitiveCompiler()

    node1 = EvidenceNode(
        node_id="n1", session_id="S1", session_date="2022-01-01",
        speaker="user", raw_text="We stayed at a luxury hotel in Rome.",
        tokens=30, entities=["Rome"], properties=["stay"], values=["hotel"]
    )
    bundle1 = EvidenceBundle("b1", "n1", [node1], 30, "anchor")

    # Call with adaptive_escalation=True
    compiled = compiler.compile(
        question="Which luxury hotel did I stay at in Rome?",
        bundles=[bundle1],
        token_budget=500,
        anchor_scores={"n1": 85.0},
        adaptive_escalation=True,
    )

    assert compiled.sufficiency_assessment is not None
    assert compiled.sufficiency_assessment.is_sufficient
