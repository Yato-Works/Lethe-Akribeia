"""Evidence Closure Engine (ECE) / Memory Compiler v2 - Type Definitions.

Strict contract:
- Deterministic, zero LLM calls during execution (Write LLM = 0).
- Verbatim text and Turn ID provenance preserved throughout.
- Explicit unfulfillment tracking: never guesses unproved requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, Any


class AnswerTopology(str, Enum):
    """Structural expectation of the target answer."""
    SINGLE_FACT = "single_fact"
    SET_ENUMERATION = "set_enumeration"
    TEMPORAL_ORDER = "temporal_order"
    TEMPORAL_DURATION = "temporal_duration"
    STATE_EVOLUTION = "state_evolution"
    INFERENCE_BOOLEAN = "inference_boolean"
    COMPARISON = "comparison"


class RequirementStatus(str, Enum):
    """Verification state of an evidence requirement."""
    UNRESOLVED = "unresolved"
    PARTIALLY_SUPPORTED = "partially_supported"
    SATISFIED = "satisfied"
    CONFLICTING = "conflicting"
    PLANNING_INCOMPLETE = "planning_incomplete"


@dataclass(frozen=True)
class EntityTarget:
    """An entity involved in the query and its coreference scope."""
    canonical_name: str
    aliases: tuple[str, ...] = ()
    coref_tokens: tuple[str, ...] = ()  # e.g. ("she", "her", "my friend", "melanie's kids")
    is_speaker: bool = False
    confidence: float = 1.0


@dataclass(frozen=True)
class PredicateRequirement:
    """A semantic relation, action, or property required by the query."""
    predicate_name: str                  # e.g. "participate_in", "paint", "made", "support"
    category: str                        # "action", "creation", "state", "origin", "relation"
    target_concept: str | None = None    # e.g. "lgbtq_event", "pottery", "art", "counseling"
    modifiers: tuple[str, ...] = ()      # e.g. ("recently", "with kids", "together")


@dataclass(frozen=True)
class TemporalConstraint:
    """Time scope or ordering boundary governing the query."""
    reference_date: str | None = None
    time_expression: str | None = None   # e.g. "recently", "last week", "4 years ago", "in June"
    relation: str = "any"                # "before", "after", "exact_date", "duration", "relative_window"
    calendar_bounds: tuple[str, str] | None = None


@dataclass(frozen=True)
class EvidenceRequirement:
    """An individual sub-goal that must be proven by dialogue evidence."""
    requirement_id: str
    description: str
    target_entity: str | None = None
    target_predicate: str | None = None
    status: RequirementStatus = RequirementStatus.UNRESOLVED
    supporting_turn_ids: tuple[str, ...] = ()
    coverage_basis: str = ""             # specific deterministic rule or facts supporting status
    unresolved_reason: str = ""          # explicit reason if requirement remains unfulfilled or conflicting
    confidence: float = 0.0
    evidence_notes: str = ""


@dataclass(frozen=True)
class QueryPlan:
    """Complete typed execution plan produced by the Query Operator Compiler."""
    query_text: str
    topology: AnswerTopology
    entity_targets: tuple[EntityTarget, ...]
    predicates: tuple[PredicateRequirement, ...]
    temporal_constraints: tuple[TemporalConstraint, ...]
    requirements: tuple[EvidenceRequirement, ...]
    recommended_budget: int
    max_closure_hops: int = 3
    exploration_threshold: float = 0.5


@dataclass(frozen=True)
class ClosureCandidate:
    """A verbatim conversation turn retrieved with provenance and multi-view scores."""
    turn_id: str                         # e.g. "D3:1" or "session_3:t0"
    session_id: str                      # e.g. "session_3"
    session_date: str                    # e.g. "9 June, 2023"
    speaker: str                         # e.g. "caroline"
    content: str                         # verbatim turn text
    view_ranks: dict[str, int] = field(default_factory=dict)
    rrf_score: float = 0.0
    satisfied_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceGraphNode:
    """Node in the verifiable evidence graph corresponding to a verbatim turn."""
    node_id: str
    turn_id: str
    session_id: str
    speaker: str
    verbatim_text: str
    extracted_predicates: tuple[str, ...] = ()
    entities_mentioned: tuple[str, ...] = ()
    timestamp_scope: str | None = None


@dataclass(frozen=True)
class EvidenceGraphEdge:
    """Verifiable link between evidence nodes."""
    source_id: str
    target_id: str
    relation_type: str                   # e.g. "DIALOGUE_TURN_CONTINUATION", "SAME_SPEAKER_REMENTION", "TEMPORAL_NEXT", "PREDICATE_CHAIN"
    confidence: float = 1.0
    provenance: str = ""


@dataclass
class ClosureState:
    """Dynamic state of the Evidence Closure loop."""
    iteration: int = 0
    satisfied_requirements: set[str] = field(default_factory=set)
    partially_satisfied_requirements: set[str] = field(default_factory=set)
    conflicting_requirements: set[str] = field(default_factory=set)
    unmet_requirements: set[str] = field(default_factory=set)
    explored_turn_ids: set[str] = field(default_factory=set)
    is_converged: bool = False
    termination_reason: str = "in_progress"
    total_tokens_accumulated: int = 0


@dataclass(frozen=True)
class EvidenceClosureResult:
    """Final output of the Evidence Closure Engine ready for Evidence Compilation."""
    query_plan: QueryPlan
    closure_state: ClosureState
    admitted_candidates: tuple[ClosureCandidate, ...]
    unmet_requirements: tuple[str, ...]
    graph_nodes: tuple[EvidenceGraphNode, ...]
    graph_edges: tuple[EvidenceGraphEdge, ...]
    closure_completeness: float          # ratio of satisfied requirements in [0.0, 1.0], 0.0 if empty plan
    requirement_details: tuple[EvidenceRequirement, ...] = ()
    provenance_summary: dict[str, Any] = field(default_factory=dict)

