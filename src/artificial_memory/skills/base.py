"""Memory Skill Framework - Base Classes and Registry (AM Apex Phase Skill Dispatcher).

Provides a clean abstraction for external deterministic co-processors that can be
invoked by the memory runtime when specialized reasoning is detected.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from artificial_memory.core.ir.structured import StructuredIR

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """Result of a skill execution.
    
    Attributes:
        skill_name: Name of the skill that produced this result.
        success: Whether the skill executed successfully.
        skill_block: Formatted text block to inject into the context (e.g., 
                     "[AM SKILL CO-PROCESSOR: TEMPORAL CALCULATOR]...").
        raw_output: Raw structured output from the skill (for debugging/logging).
        metadata: Additional metadata about the skill execution.
        error: Error message if success=False.
    """
    skill_name: str
    success: bool
    skill_block: str = ""
    raw_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success


class MemorySkill(abc.ABC):
    """Abstract base class for all memory skills (deterministic co-processors).
    
    A MemorySkill is a deterministic Python co-processor that handles specialized
    reasoning tasks that LLMs struggle with (e.g., calendar arithmetic, entity
    deduplication, cross-session counting). Skills are invoked by the SkillDispatcher
    when the query matches the skill's domain.
    
    Key principles:
    - 100% deterministic: same input always produces same output
    - No LLM calls: pure Python logic
    - Zero cost at write time: invoked at query time only
    - Provides a skill_block that can be injected into the compiled context
    """
    
    # Class-level identifier for the skill
    name: str = "base_skill"
    
    # Human-readable description of what this skill handles
    description: str = "Base memory skill"
    
    # Keywords/patterns that indicate this skill should be considered
    trigger_keywords: list[str] = []
    
    # Question categories (LoCoMo) this skill handles
    target_categories: list[int] = []
    
    # LongMemEval question types this skill handles
    target_question_types: list[str] = []
    
    def __init__(self) -> None:
        self._invocation_count = 0
        self._success_count = 0
    
    @abc.abstractmethod
    def can_handle(self, question: str, records: Sequence[StructuredIR], 
                   category: Optional[int] = None, 
                   question_type: Optional[str] = None) -> bool:
        """Determine if this skill can handle the given query.
        
        Args:
            question: The user's question text.
            records: StructuredIR records from the conversation.
            category: LoCoMo category (1-5) if applicable.
            question_type: LongMemEval question type if applicable.
            
        Returns:
            True if this skill should be invoked for this query.
        """
        pass
    
    @abc.abstractmethod
    def resolve(self, question: str, records: Sequence[StructuredIR],
                reference_date: Optional[str] = None,
                **kwargs) -> "SkillResult":
        """Execute the skill and return a formatted result.
        
        Args:
            question: The user's question text.
            records: StructuredIR records from the conversation.
            reference_date: Optional reference date for temporal queries.
            **kwargs: Additional context (e.g., intent, query plan).
            
        Returns:
            SkillResult with formatted skill_block for context injection.
        """
        pass
    
    def _make_result(self, success: bool, skill_block: str = "",
                     raw_output: Any = None, metadata: Optional[dict] = None,
                     error: Optional[str] = None) -> SkillResult:
        """Helper to create a SkillResult."""
        self._invocation_count += 1
        if success:
            self._success_count += 1
        return SkillResult(
            skill_name=self.name,
            success=success,
            skill_block=skill_block,
            raw_output=raw_output,
            metadata=metadata or {},
            error=error
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Return invocation statistics."""
        return {
            "name": self.name,
            "invocations": self._invocation_count,
            "successes": self._success_count,
            "success_rate": self._success_count / max(1, self._invocation_count)
        }
    
    def reset_stats(self) -> None:
        self._invocation_count = 0
        self._success_count = 0


class SkillRegistry:
    """Registry for managing and dispatching memory skills.
    
    Maintains a collection of skills and provides dispatch logic to find
    and invoke the appropriate skill for a given query.
    """
    
    def __init__(self) -> None:
        self._skills: dict[str, MemorySkill] = {}
        self._skill_order: list[str] = []
    
    def register(self, skill: MemorySkill) -> None:
        """Register a skill in the registry."""
        if skill.name in self._skills:
            logger.warning(f"Overwriting existing skill: {skill.name}")
        self._skills[skill.name] = skill
        if skill.name not in self._skill_order:
            self._skill_order.append(skill.name)
        logger.info(f"Registered skill: {skill.name} - {skill.description}")
    
    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry."""
        if name in self._skills:
            del self._skills[name]
            self._skill_order.remove(name)
            return True
        return False
    
    def get(self, name: str) -> Optional[MemorySkill]:
        """Get a skill by name."""
        return self._skills.get(name)
    
    def find_skills(self, question: str, records: Sequence[StructuredIR],
                    category: Optional[int] = None,
                    question_type: Optional[str] = None) -> list[MemorySkill]:
        """Find all skills that can handle the given query.
        
        Skills are returned in registration order (priority order).
        """
        applicable = []
        for name in self._skill_order:
            skill = self._skills[name]
            if skill.can_handle(question, records, category, question_type):
                applicable.append(skill)
        return applicable
    
    def dispatch(self, question: str, records: Sequence[StructuredIR],
                 category: Optional[int] = None,
                 question_type: Optional[str] = None,
                 reference_date: Optional[str] = None,
                 **kwargs) -> list[SkillResult]:
        """Find and execute all applicable skills for the query.
        
        Returns a list of SkillResult objects, one per applicable skill.
        """
        skills = self.find_skills(question, records, category, question_type)
        results = []
        for skill in skills:
            try:
                result = skill.resolve(question, records, reference_date, **kwargs)
                results.append(result)
                if result.success and result.skill_block:
                    logger.info(f"Skill {skill.name} executed successfully")
                elif not result.success:
                    logger.warning(f"Skill {skill.name} failed: {result.error}")
            except Exception as e:
                logger.exception(f"Skill {skill.name} raised exception: {e}")
                results.append(SkillResult(
                    skill_name=skill.name,
                    success=False,
                    error=str(e)
                ))
        return results
    
    def get_all_stats(self) -> dict[str, dict]:
        return {name: skill.get_stats() for name, skill in self._skills.items()}


# Global registry instance
_global_registry = SkillRegistry()


def get_global_registry() -> SkillRegistry:
    """Get the global skill registry instance."""
    return _global_registry


def register_skill(skill: MemorySkill) -> None:
    """Convenience function to register a skill globally."""
    _global_registry.register(skill)


def dispatch_skills(question: str, records: Sequence[StructuredIR],
                    category: Optional[int] = None,
                    question_type: Optional[str] = None,
                    reference_date: Optional[str] = None,
                    **kwargs) -> list[SkillResult]:
    """Convenience function to dispatch skills globally."""
    return _global_registry.dispatch(question, records, category, question_type, reference_date, **kwargs)


# Convenience exports
__all__ = [
    "SkillResult",
    "MemorySkill",
    "SkillRegistry",
    "get_global_registry",
    "register_skill",
    "dispatch_skills",
]