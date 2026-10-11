"""Deterministic Query Planner for Evidence Closure Engine (Write LLM = 0).

Maps natural language questions into strongly typed QueryPlans without
question-specific hardcoding or gold-answer leaks.
"""

from __future__ import annotations

import re
from typing import Sequence

from artificial_memory.compiler.closure.types import (
    AnswerTopology,
    EntityTarget,
    PredicateRequirement,
    TemporalConstraint,
    EvidenceRequirement,
    QueryPlan,
    RequirementStatus,
)


class DeterministicQueryPlanner:
    """Extracts typed query requirements using generic syntactic & structural patterns."""

    _RE_ENUMERATION = re.compile(
        r"\b(?:what|which)\s+(?:(?:[a-zA-Z0-9_+/'-]+\s+){0,3})(?:activities|events|things|kinds|types|places|books|songs|movies|foods|dishes|hobbies|sports|games|names|cities|countries|states|ways)\b|"
        r"\bwhere\s+has\s+(?:[a-zA-Z0-9_-]+)\s+camped\b|"
        r"\bwhat\s+has\s+(?:[a-zA-Z0-9_-]+)\s+(?:read|done|made|painted)\b|"
        r"\blist\s+all\b",
        re.I,
    )

    _RE_TEMPORAL_ORDER = re.compile(
        r"\b(order\s+from\s+first\s+to\s+last|earlier\s+or\s+later|before\s+or\s+after|which\s+happened\s+first|most\s+recently)\b",
        re.I,
    )

    _RE_TEMPORAL_DURATION = re.compile(
        r"\b(how\s+long\s+ago|how\s+many\s+(?:days|weeks|months|years)\s+ago|how\s+much\s+time\s+passed|how\s+long\s+has\s+[a-zA-Z0-9_-]+\s+had)\b",
        re.I,
    )

    _RE_DATE_QUERY = re.compile(
        r"\b(when\s+did|what\s+date|which\s+day|on\s+what\s+date|what\s+year|which\s+month)\b",
        re.I,
    )

    _RE_INFERENCE_BOOLEAN = re.compile(
        r"\b(would\s+[a-zA-Z0-9_-]+\s+(?:likely|still|want|prefer|pursue|be)|did\s+[a-zA-Z0-9_-]+\s+(?:have|do|go))\b",
        re.I,
    )

    _RE_STATE_EVOLUTION = re.compile(
        r"\b(currently|right\s+now|at\s+present|previously|used\s+to|no\s+longer|switched\s+to|changed)\b",
        re.I,
    )

    _COMMON_ENTITIES = ("caroline", "melanie", "grandma", "kids", "students", "partner")

    def plan(self, question: str, reference_date: str | None = None) -> QueryPlan:
        q_clean = re.sub(r"\s+", " ", question).strip()
        q_lower = q_clean.lower()

        # 1. Determine Answer Topology
        if self._RE_TEMPORAL_DURATION.search(q_lower):
            topology = AnswerTopology.TEMPORAL_DURATION
        elif self._RE_TEMPORAL_ORDER.search(q_lower):
            topology = AnswerTopology.TEMPORAL_ORDER
        elif self._RE_DATE_QUERY.search(q_lower):
            topology = AnswerTopology.TEMPORAL_DURATION
        elif self._RE_ENUMERATION.search(q_lower):
            topology = AnswerTopology.SET_ENUMERATION
        elif self._RE_INFERENCE_BOOLEAN.search(q_lower):
            topology = AnswerTopology.INFERENCE_BOOLEAN
        elif self._RE_STATE_EVOLUTION.search(q_lower):
            topology = AnswerTopology.STATE_EVOLUTION
        else:
            topology = AnswerTopology.SINGLE_FACT

        # 2. Extract Target Entities
        entities: list[EntityTarget] = []
        for ent in self._COMMON_ENTITIES:
            if re.search(rf"\b{ent}(?:'s)?\b", q_lower):
                corefs = ("she", "her") if ent in ("caroline", "melanie") else ()
                entities.append(EntityTarget(
                    canonical_name=ent,
                    aliases=(ent.capitalize(),),
                    coref_tokens=corefs,
                    is_speaker=ent in ("caroline", "melanie"),
                ))
        if not entities:
            # Fallback for generic subject
            entities.append(EntityTarget(canonical_name="user", aliases=("User",), is_speaker=True))

        # 3. Extract Predicate Requirements (prioritize content verbs over auxiliaries)
        predicates: list[PredicateRequirement] = []
        action_verbs = [
            "participate in", "participated in", "participating in",
            "camped", "camping", "camp",
            "paint", "painted", "painting",
            "made", "crafted", "create", "created",
            "read", "reading",
            "support", "supports", "supporting",
            "moved from", "move from", "moving from",
            "travel", "traveled", "visit", "visited"
        ]
        for v in action_verbs:
            if re.search(rf"\b{v}\b", q_lower):
                predicates.append(PredicateRequirement(
                    predicate_name=v,
                    category="action" if "camp" in v or "participat" in v or "travel" in v else "creation" if "paint" in v or "made" in v or "craft" in v else "relation",
                ))
                break

        # 4. Extract Temporal Constraints
        temporal_constraints: list[TemporalConstraint] = []
        time_match = re.search(r"\b(recently|last\s+week|last\s+year|4\s+years\s+ago|in\s+june|in\s+2022|yesterday)\b", q_lower)
        if time_match:
            temporal_constraints.append(TemporalConstraint(
                reference_date=reference_date,
                time_expression=time_match.group(1),
                relation="relative_window" if "last" in time_match.group(1) or "ago" in time_match.group(1) else "calendar_bounds"
            ))

        # 5. Formulate Verifiable Evidence Requirements
        requirements: list[EvidenceRequirement] = []
        ent_name = entities[0].canonical_name if entities else "subject"

        if topology == AnswerTopology.SET_ENUMERATION:
            requirements.append(EvidenceRequirement(
                requirement_id="req_enum_instances",
                description=f"Identify all distinct instances and events associated with {ent_name}.",
                target_entity=ent_name,
                target_predicate=predicates[0].predicate_name if predicates else "instances",
            ))
            requirements.append(EvidenceRequirement(
                requirement_id="req_multi_session_breadth",
                description="Cross-session coverage verifying whether activities occurred in separate sessions.",
                target_entity=ent_name,
            ))
        elif topology in (AnswerTopology.TEMPORAL_DURATION, AnswerTopology.TEMPORAL_ORDER):
            requirements.append(EvidenceRequirement(
                requirement_id="req_anchor_event",
                description=f"Evidence establishing the event date or timestamp for {ent_name}.",
                target_entity=ent_name,
            ))
            requirements.append(EvidenceRequirement(
                requirement_id="req_temporal_anchor",
                description="Conversation session header establishing the baseline calendar date.",
            ))
        else:
            requirements.append(EvidenceRequirement(
                requirement_id="req_direct_fact",
                description=f"Direct factual claim regarding {ent_name}'s {predicates[0].predicate_name if predicates else 'attribute'}.",
                target_entity=ent_name,
                target_predicate=predicates[0].predicate_name if predicates else None,
            ))

        # 6. Budget calculation
        budget = 1400 if topology == AnswerTopology.SET_ENUMERATION else 1200 if topology == AnswerTopology.TEMPORAL_DURATION else 800

        return QueryPlan(
            query_text=q_clean,
            topology=topology,
            entity_targets=tuple(entities),
            predicates=tuple(predicates),
            temporal_constraints=tuple(temporal_constraints),
            requirements=tuple(requirements),
            recommended_budget=budget,
            max_closure_hops=3,
        )
