"""Evidence Coverage Planner for Multi-Session Ingestion.

Analyzes query intent (COUNT, SUM, DIFFERENCE, ENUMERATION, PAIRWISE) and
plans the structural evidence requirements across conversational memory sessions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from artificial_memory_gen.compiler.coverage.concept_expander import AbstractConceptExpander
from artificial_memory_gen.compiler.coverage.self_anchor_expander import (
    PairwiseEvidenceRequirement,
    SelfAnchorExpander,
)


@dataclass(frozen=True)
class PlannedEvidenceRequirements:
    """The structured evidence requirements for satisfying a query."""
    intent: str                                       # 'COUNT', 'SUM', 'DIFFERENCE', 'LIST', 'SINGLE_FACT'
    primary_query: str
    concept_subqueries: list[str]                     # Subqueries derived from category expansion
    pairwise_requirements: list[PairwiseEvidenceRequirement]  # Dual entity or self-anchor requirements
    minimum_expected_sessions: int                    # Lower bound of sessions needed for full coverage


class EvidenceCoveragePlanner:
    """Plans memory evidence requirements based on query semantics."""

    _RE_COUNT = re.compile(r"\bhow many\b", re.I)
    _RE_SUM = re.compile(r"\b(?:total amount|total cost|total number|total hours|sum|altogether|combined)\b", re.I)
    _RE_DIFF = re.compile(r"\b(?:difference|how much more|how much less|older than|younger than|faster than|slower than|compared to)\b", re.I)
    _RE_LIST = re.compile(r"\b(?:which|what|list|name all)\b", re.I)

    @classmethod
    def plan(cls, query: str) -> PlannedEvidenceRequirements:
        """Analyze query and establish evidence requirements."""
        q_lower = query.lower()

        # 1. Determine Intent
        if cls._RE_DIFF.search(q_lower):
            intent = "DIFFERENCE"
            min_sessions = 2
        elif cls._RE_SUM.search(q_lower):
            intent = "SUM"
            min_sessions = 2
        elif cls._RE_COUNT.search(q_lower):
            intent = "COUNT"
            min_sessions = 2
        elif cls._RE_LIST.search(q_lower):
            intent = "LIST"
            min_sessions = 1
        else:
            intent = "SINGLE_FACT"
            min_sessions = 1

        # 2. Extract Concept Subqueries
        concept_subs = AbstractConceptExpander.generate_subqueries(query)

        # 3. Extract Pairwise Requirements
        pair_reqs = SelfAnchorExpander.analyze(query)
        if pair_reqs:
            min_sessions = max(min_sessions, 2)

        return PlannedEvidenceRequirements(
            intent=intent,
            primary_query=query,
            concept_subqueries=concept_subs,
            pairwise_requirements=pair_reqs,
            minimum_expected_sessions=min_sessions,
        )
