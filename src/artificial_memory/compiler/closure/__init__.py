"""Evidence Closure Engine (ECE) / Memory Compiler v2."""

from artificial_memory.compiler.closure.types import (
    AnswerTopology,
    RequirementStatus,
    EntityTarget,
    PredicateRequirement,
    TemporalConstraint,
    EvidenceRequirement,
    QueryPlan,
    ClosureCandidate,
    EvidenceGraphNode,
    EvidenceGraphEdge,
    ClosureState,
    EvidenceClosureResult,
)
from artificial_memory.compiler.closure.planner import DeterministicQueryPlanner
from artificial_memory.compiler.closure.retriever import MultiViewRetriever
from artificial_memory.compiler.closure.closure_engine import EvidenceClosureEngine
from artificial_memory.compiler.closure.compiler_v2 import CompiledContextV2, EvidenceCompilerV2

__all__ = [
    "AnswerTopology",
    "RequirementStatus",
    "EntityTarget",
    "PredicateRequirement",
    "TemporalConstraint",
    "EvidenceRequirement",
    "QueryPlan",
    "ClosureCandidate",
    "EvidenceGraphNode",
    "EvidenceGraphEdge",
    "ClosureState",
    "EvidenceClosureResult",
    "DeterministicQueryPlanner",
    "MultiViewRetriever",
    "EvidenceClosureEngine",
    "CompiledContextV2",
    "EvidenceCompilerV2",
]


