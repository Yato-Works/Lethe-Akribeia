"""AM Memory Skills - Deterministic Co-processor Framework.

This module provides the core skill framework for AM Apex, enabling
deterministic Python co-processors to handle specialized reasoning tasks
that LLMs struggle with (calendar arithmetic, entity deduplication, etc.).

Usage:
    from artificial_memory.skills import TemporalSkill, get_global_registry
    
    registry = get_global_registry()
    results = dispatch_skills(question, records, category=2)
"""

from __future__ import annotations

from .aggregation_skill import AggregationCalculation, AggregationSkill
from .base import (
    MemorySkill,
    SkillRegistry,
    SkillResult,
    dispatch_skills,
    get_global_registry,
    register_skill,
)
from .temporal_skill import TemporalCalculation, TemporalSkill

# Lazy import to avoid circular dependencies
_temporal_skill_instance = None
_aggregation_skill_instance = None


def get_temporal_skill() -> TemporalSkill:
    """Get or create the global TemporalSkill instance."""
    global _temporal_skill_instance
    if _temporal_skill_instance is None:
        from .temporal_skill import TemporalSkill
        _temporal_skill_instance = TemporalSkill()
        register_skill(_temporal_skill_instance)
    return _temporal_skill_instance


def get_aggregation_skill() -> AggregationSkill:
    """Get or create the global AggregationSkill instance."""
    global _aggregation_skill_instance
    if _aggregation_skill_instance is None:
        from .aggregation_skill import AggregationSkill
        _aggregation_skill_instance = AggregationSkill()
        register_skill(_aggregation_skill_instance)
    return _aggregation_skill_instance


__all__ = [
    "SkillResult",
    "MemorySkill",
    "SkillRegistry",
    "get_global_registry",
    "register_skill",
    "dispatch_skills",
    "get_temporal_skill",
    "TemporalSkill",
    "TemporalCalculation",
    "get_aggregation_skill",
    "AggregationSkill",
    "AggregationCalculation",
]
