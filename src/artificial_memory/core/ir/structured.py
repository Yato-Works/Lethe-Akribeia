"""Universal Structured Cognitive Intermediate Representation (IR).

Phase 2: Replaces all heuristic string checks with a generalized
Entity-Property-Value-Time-Source-Relation IR architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class IRRelation(StrEnum):
    """Semantic relation types between entity and value."""
    ASSERTS = "asserts"              # Default factual assertion (X has Y)
    PREFERS = "prefers"              # Subjective preference (X prefers Y)
    MIGRATED = "migrated"            # Transition / update (moved from X to Y)
    LOCATED_AT = "located_at"        # Physical or virtual location
    NEVER = "never"                  # Explicit negative constraint (never X)
    CONTRADICTS = "contradicts"      # Explicit conflicting statements
    SUPPORTS = "supports"            # Corroborating evidence
    BEHAVIOR = "behavior"            # Behavioral tendency (e.g., reverts to good state)


class IRStatus(StrEnum):
    """Lifecycle status of a structured memory representation."""
    ACTIVE = "active"                # Currently valid and asserted
    SUPERSEDED = "superseded"        # Replaced by a newer value (e.g. migration)
    UNRESOLVED_CONFLICT = "unresolved_conflict"  # Conflicting reports between sources
    DEPRECATED = "deprecated"        # Explicitly marked as old/inactive
    UNSPECIFIED = "unspecified"      # Known to be not specified / never set


@dataclass
class StructuredIR:
    """Universal Structured Cognitive Intermediate Representation (IR).

    Encapsulates knowledge extracted from dialogue turns into a queryable,
    verifiable, and traceable format.
    """
    entity: str                      # Subject / Scope (e.g. "project atlas", "Eli", "workstation")
    property: str                    # Attribute / Aspect (e.g. "database", "location", "debugger", "drinks")
    value: str                       # Current or asserted value (e.g. "ClickHouse", "Kyoto", "matcha")
    old_value: str | None = None     # Prior value when updated (e.g. "SQLite")
    time_scope: str | None = None    # Temporal qualification (e.g. "until 2026-02-03", "2026-01-27")
    valid_from: str | None = None    # When the fact became true in the world (e.g. "2023-01-01")
    valid_until: str | None = None   # When the fact ceased to be true (None if still active)
    assertion_time: str | None = None # When the fact was uttered/recorded in dialogue
    source: str = "user"             # Provenance / Speaker ("user", "teammate", "assistant")
    relation: IRRelation = IRRelation.ASSERTS
    status: IRStatus = IRStatus.ACTIVE
    confidence: float = 1.0          # Subjective confidence [0.0, 1.0]
    raw_content: str = ""            # Original verbatim text
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def is_valid_at(self, target_date_str: str) -> bool:
        """Check if this fact was valid at a specific point in time (Time-Travel query)."""
        if not self.valid_from and not self.valid_until:
            return True
        if self.valid_from and target_date_str < self.valid_from:
            return False
        if self.valid_until and target_date_str > self.valid_until:
            return False
        return True

    def supersede(self, new_value: str, superseded_at: str) -> None:
        """Mark this state as superseded by a newer fact as of a specific date."""
        self.old_value = self.value
        self.value = new_value
        self.valid_until = superseded_at
        self.status = IRStatus.SUPERSEDED
        self.relation = IRRelation.MIGRATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "property": self.property,
            "value": self.value,
            "old_value": self.old_value,
            "time_scope": self.time_scope,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "assertion_time": self.assertion_time,
            "source": self.source,
            "relation": self.relation.value,
            "status": self.status.value,
            "confidence": self.confidence,
            "raw_content": self.raw_content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    def format_context_line(self) -> str:
        """Format as a clear, natural context line for LLM consumption."""
        if self.status == IRStatus.UNRESOLVED_CONFLICT:
            return f"Conflict for {self.entity} {self.property}: {self.source} reported {self.value}, but this remains an unresolved conflict."
        if self.status == IRStatus.SUPERSEDED and self.old_value:
            until_str = f" until {self.valid_until}" if self.valid_until else ""
            return f"For {self.entity}, {self.property} was updated: previously {self.old_value}{until_str}, currently {self.value}."
        if self.relation == IRRelation.MIGRATED and self.old_value:
            return f"For {self.entity}, {self.property} was migrated from {self.old_value} to {self.value}."
        if self.relation == IRRelation.NEVER:
            return f"For {self.entity}, {self.property} was never configured or used."
        if self.time_scope:
            return f"For {self.entity}, {self.property} is {self.value} ({self.time_scope})."
        return f"For {self.entity}, {self.property} is {self.value} (source: {self.source})."
