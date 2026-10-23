"""Evidence Closure Engine (ECE) / Memory Compiler v2."""

from artificial_memory_gen.compiler.closure.closure_engine import EvidenceClosureEngine
from artificial_memory_gen.compiler.closure.compiler_v2 import CompiledContextV2, EvidenceCompilerV2
from artificial_memory_gen.compiler.closure.planner import DeterministicQueryPlanner
from artificial_memory_gen.compiler.closure.retriever import MultiViewRetriever
from artificial_memory_gen.compiler.closure.types import (
    AnswerTopology,
    ClosureCandidate,
    ClosureState,
    EntityTarget,
    EvidenceClosureResult,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    EvidenceRequirement,
    PredicateRequirement,
    QueryPlan,
    RequirementStatus,
    TemporalConstraint,
)

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


