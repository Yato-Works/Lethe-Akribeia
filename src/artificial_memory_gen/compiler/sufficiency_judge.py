"""Evidence Sufficiency Judge & Adaptive Escalation Engine (R6).

Evaluates whether retrieved and compiled evidence meets sufficiency criteria for answering,
using deterministic state features extracted from the Execution Planner, Dependency Graph,
and Submodular Knapsack Compiler.

Trigger conditions for Adaptive Escalation:
1. Low Coverage & Low Budget Utilization (Budget starvation)
2. Weak Anchor Confidence (Top anchor score < 55.0 and coverage < 0.40)
3. Multi-Session Under-Representation (Multi-session query with <= 1 session selected)
4. Vocabulary Mismatch (High word count query with minimal lexical overlap)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SufficiencyAssessment:
    """The judgment result of evidence sufficiency."""
    is_sufficient: bool
    sufficiency_score: float             # 0.0 (severely insufficient) to 1.0 (fully saturated)
    distress_reasons: list[str]          # Identifiers of why evidence is considered insufficient
    recommended_escalations: list[str]   # Actions to escalate: 'expand_sessions', 'anchor_prefill', 'relax_density', 'boost_budget'
    budget_escalation_factor: float      # Multiplier for token budget if starvation detected (e.g. 1.25 - 1.5)


class EvidenceSufficiencyJudge:
    """Deterministic, zero-LLM evidence sufficiency evaluator for memory compilation."""

    # Separation thresholds derived from R6 cross-condition feature distributions
    MIN_HEALTHY_COVERAGE = 0.35
    MIN_HEALTHY_BUDGET_UTIL = 0.65
    MIN_HEALTHY_ANCHOR_SCORE = 55.0

    def evaluate(
        self,
        query: str,
        plan,
        compiled_context,
        top1_anchor_score: float,
        selected_session_count: int,
        is_multi_session: bool,
        candidate_count: int = 0,
    ) -> SufficiencyAssessment:
        """Evaluate evidence sufficiency and recommend adaptive escalation actions."""
        distress: list[str] = []
        escalations: list[str] = []
        budget_factor = 1.0

        if compiled_context is None:
            return SufficiencyAssessment(
                is_sufficient=False,
                sufficiency_score=0.0,
                distress_reasons=["null_compiled_context"],
                recommended_escalations=["expand_sessions", "anchor_prefill", "boost_budget"],
                budget_escalation_factor=1.5,
            )

        # 1. Evidence Coverage
        q_clean = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))
        q_len = max(1, len(q_clean))
        coverage = min(1.0, float(compiled_context.utility_score) / q_len)

        # 2. Budget Utilization
        budget_limit = max(1, compiled_context.budget_limit)
        budget_util = min(1.0, float(compiled_context.total_tokens) / budget_limit)

        # 3. Anchor Score Ratio
        anchor_ratio = min(1.0, max(0.0, top1_anchor_score / 75.0))

        # Composite sufficiency score
        # Higher coverage, healthy budget util, and strong anchors contribute to confidence
        score = (coverage * 0.40) + (budget_util * 0.35) + (anchor_ratio * 0.25)

        # High-confidence evidence override:
        # When we have strong anchor confidence (>= 75.0) and high query coverage (>= 50%),
        # or top1 anchor is extremely confident (>= 80.0), a concise turn is completely sufficient.
        is_strong_evidence = (top1_anchor_score >= 70.0 and coverage >= 0.50) or (top1_anchor_score >= 80.0)

        # ─── Distress Condition 1: Budget Starvation & Low Coverage ───
        # When budget is underutilized (< 65%) and query coverage is poor (< 35%),
        # AND we don't already have high-confidence evidence
        if not is_strong_evidence and budget_util < self.MIN_HEALTHY_BUDGET_UTIL and coverage < self.MIN_HEALTHY_COVERAGE:
            distress.append("budget_starvation")
            escalations.append("relax_density")
            escalations.append("anchor_prefill")

        # ─── Distress Condition 2: Weak Anchor Confidence ───
        # Low top-1 anchor score (< 55.0) combined with low coverage
        if top1_anchor_score < self.MIN_HEALTHY_ANCHOR_SCORE and coverage < 0.40:
            distress.append("weak_anchor_signal")
            escalations.append("expand_sessions")
            escalations.append("expand_semantic_expansions")

        # ─── Distress Condition 3: Multi-Session Under-Representation ───
        # Multi-session/aggregation query where <= 1 session was captured
        if is_multi_session and selected_session_count <= 1:
            distress.append("multi_session_under_represented")
            escalations.append("expand_sessions")
            escalations.append("boost_budget")
            budget_factor = max(budget_factor, 1.35)

        # ─── Distress Condition 4: Candidate Turn Starvation ───
        # Only triggers when candidate pool was large (>= 8) but coverage is poor (< 35%)
        if not is_strong_evidence and candidate_count >= 8 and len(compiled_context.selected_node_ids) <= 2 and coverage < 0.35:
            distress.append("candidate_turn_starvation")
            escalations.append("relax_density")

        is_sufficient = len(distress) == 0 and (score >= 0.40 or is_strong_evidence)

        # De-duplicate escalations while preserving order
        dedup_escalations = list(dict.fromkeys(escalations))

        return SufficiencyAssessment(
            is_sufficient=is_sufficient,
            sufficiency_score=round(score, 4),
            distress_reasons=distress,
            recommended_escalations=dedup_escalations,
            budget_escalation_factor=budget_factor,
        )
