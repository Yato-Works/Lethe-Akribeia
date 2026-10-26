"""Legacy skill stack (pre-DME), retained for research benchmark adapters.

The mainline runtime uses
``artificial_memory.skills.DeterministicMemoryEngine``; this package keeps the
older ``StructuredIR``-based ``TemporalSkill`` / ``AggregationSkill`` stack
that ``research/benchmarks/external/*_adapter.py`` still route through.
"""

from __future__ import annotations

from .aggregation_skill import AggregationCalculation, AggregationSkill
from .base import (
    MemorySkill,
    SkillRegistry,
    SkillResult,
    dispatch_skills,
    register_skill,
)
from .temporal_skill import TemporalCalculation, TemporalSkill

_skill_registry: SkillRegistry | None = None


def _registry() -> SkillRegistry:
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
        _skill_registry.register(TemporalSkill())
        _skill_registry.register(AggregationSkill())
    return _skill_registry


def get_temporal_skill() -> TemporalSkill:
    """Return the shared TemporalSkill instance (research adapters)."""
    _registry()
    return TemporalSkill()


def get_aggregation_skill() -> AggregationSkill:
    """Return the shared AggregationSkill instance (research adapters)."""
    _registry()
    return AggregationSkill()


__all__ = [
    "MemorySkill",
    "SkillRegistry",
    "SkillResult",
    "register_skill",
    "dispatch_skills",
    "get_temporal_skill",
    "TemporalSkill",
    "TemporalCalculation",
    "get_aggregation_skill",
    "AggregationSkill",
    "AggregationCalculation",
]
