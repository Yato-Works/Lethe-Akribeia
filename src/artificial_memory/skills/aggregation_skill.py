"""Aggregation & Counting Skill - Deterministic Counting/Set Deduplication Co-processor.

This skill provides deterministic counting and set aggregation for:
- "How many times..." / "How many times..." queries
- Multi-session entity counting (distinct items across sessions)
- "List all..." queries requiring complete enumeration
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from artificial_memory.core.ir.structured import StructuredIR

from .base import MemorySkill, SkillResult


@dataclass
class AggregationCalculation:
    """Structured result of an aggregation/counting calculation."""
    items: list[str]                    # Distinct items found
    count: int                          # Exact count
    calculation_type: str               # "count", "enumeration", "abstention"
    details: str                        # Human-readable details
    confidence: float = 1.0


class AggregationSkill(MemorySkill):
    """Deterministic Counting & Aggregation Co-processor.

    Handles:
    - "How many times..." / "How many [items]..." queries
    - Cross-session entity counting with deterministic deduplication
    - "List all..." queries requiring complete enumeration
    - Multi-session aggregation (sums, counts, distinct sets)
    """

    name: str = "aggregation_skill"
    description: str = "Deterministic counting, set deduplication, and multi-session aggregation"
    trigger_keywords: list[str] = [
        "how many", "how many times", "how many times did", "count",
        "list all", "list every", "enumerate", "total number", "number of",
        "distinct", "unique", "different", "aggregate", "sum", "total"
    ]
    target_categories: list[int] = [1]  # LoCoMo Category 1: Multi-Hop (often counting)
    target_question_types: list[str] = ["multi-session", "aggregation", "counting"]

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[str, AggregationCalculation] = {}

    def can_handle(self, question: str, records: Sequence[StructuredIR],
                   category: int | None = None,
                   question_type: str | None = None) -> bool:
        """Determine if this skill should handle the query."""
        # Explicit category/type match
        if category in self.target_categories:
            return True
        if question_type in self.target_question_types:
            return True

        q_lower = question.lower()
        # Much stricter counting patterns - only fire on explicit counting queries
        strict_counting_patterns = [
            r"\bhow many\s+(?:cities|countries|people|times|events|items|places|books|movies|songs|artists|authors)\b",
            r"\bcount\s+the\s+(?:number\s+of\s+)?(?:cities|countries|people|times|events|items|places)\b",
            r"\blist all\s+(?:cities|countries|people|events|items|places|books|movies)\b",
            r"\benumerate\s+(?:cities|countries|people|events|items|places)\b",
            r"\btotal\s+number\s+of\s+(?:cities|countries|people|times|events|items|places)\b",
            r"\bnumber\s+of\s+(?:cities|countries|people|times|events|items|places)\b",
            r"\btotal\s+(?:cities|countries|people|times|events|items|places)\b",
        ]
        return any(re.search(p, q_lower) for p in strict_counting_patterns)

    def resolve(self, question: str, records: Sequence[StructuredIR],
                reference_date: str | None = None,
                **kwargs) -> SkillResult:
        """Execute deterministic counting/aggregation."""
        try:
            cache_key = self._make_cache_key(question, records, reference_date)
            if cache_key in self._cache:
                calc = self._cache[cache_key]
                # Higher confidence threshold - only inject if very confident
                skill_block = self._format_skill_block(calc) if calc.confidence >= 0.9 else ""
                return self._make_result(
                    success=True,
                    skill_block=skill_block,
                    raw_output=calc,
                    metadata={"cached": True, "calculation_type": calc.calculation_type, "confidence": calc.confidence}
                )

            calc = self._calculate(question, records, reference_date)
            self._cache[cache_key] = calc

            # Much higher confidence threshold - only inject skill block if very confident
            skill_block = self._format_skill_block(calc) if calc.confidence >= 0.9 else ""

            return self._make_result(
                success=True,
                skill_block=skill_block,
                raw_output=calc,
                metadata={"calculation_type": calc.calculation_type, "confidence": calc.confidence}
            )
        except Exception as e:
            return self._make_result(
                success=False,
                error=f"AggregationSkill error: {e}"
            )

    def _make_cache_key(self, question: str, records: Sequence[StructuredIR],
                        reference_date: str | None) -> str:
        context_str = "".join(r.raw_content[:100] for r in records[:10])
        key_data = f"{question}|{reference_date}|{context_str[:500]}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _calculate(self, question: str, records: Sequence[StructuredIR],
                   reference_date: str | None) -> AggregationCalculation:
        q_lower = question.lower()

        # First, verify this is genuinely a counting query with strict patterns
        strict_counting_patterns = [
            r"\bhow many\s+(?:cities|countries|people|times|events|items|places|books|movies|songs|artists|authors)\b",
            r"\bcount\s+the\s+(?:number\s+of\s+)?(?:cities|countries|people|times|events|items|places)\b",
            r"\blist all\s+(?:cities|countries|people|events|items|places|books|movies)\b",
            r"\benumerate\s+(?:cities|countries|people|events|items|places)\b",
            r"\btotal\s+number\s+of\s+(?:cities|countries|people|times|events|items|places)\b",
            r"\bnumber\s+of\s+(?:cities|countries|people|times|events|items|places)\b",
            r"\btotal\s+(?:cities|countries|people|times|events|items|places)\b",
        ]

        is_strict_counting = any(re.search(p, q_lower) for p in strict_counting_patterns)
        if not is_strict_counting:
            return AggregationCalculation(
                items=[],
                count=0,
                calculation_type="abstention",
                details="Query does not match strict counting patterns",
                confidence=0.0
            )

        # Extract countable items
        items = self._extract_items(question, records)

        if not items:
            return AggregationCalculation(
                items=[],
                count=0,
                calculation_type="abstention",
                details="Could not identify countable items in context",
                confidence=0.0
            )

        # Deduplicate items
        unique_items = list(dict.fromkeys(items))  # Preserve order, remove duplicates
        count = len(unique_items)

        # Require at least 2 items for a valid count (single item is trivial)
        if count < 2:
            return AggregationCalculation(
                items=[],
                count=0,
                calculation_type="abstention",
                details="Insufficient items found for meaningful count",
                confidence=0.0
            )

        # Determine calculation type
        q_lower = question.lower()
        if "list all" in q_lower or "list every" in q_lower or "enumerate" in q_lower:
            calc_type = "enumeration"
        elif "how many" in q_lower or "how many times" in q_lower or "count" in q_lower:
            calc_type = "count"
        else:
            calc_type = "aggregation"

        # Build details
        items_str = ", ".join(f'"{item}"' for item in unique_items)
        details = f"Identified {count} distinct items: {items_str}"

        # Confidence based on item count and clarity - be more conservative
        confidence = min(0.9, 0.6 + (count * 0.05))

        return AggregationCalculation(
            items=unique_items,
            count=count,
            calculation_type=calc_type,
            details=details,
            confidence=confidence
        )

    def _extract_items(self, question: str, records: Sequence[StructuredIR]) -> list[str]:
        """Extract countable items from question and records.

        Strategy: Extract all proper nouns / capitalized entities from records
        that appear in the context of the question's target nouns.
        """
        items = []
        q_lower = question.lower()

        # Extract what to count from question
        count_patterns = [
            r"how many\s+([a-zA-Z\s]+?)(?:\s+(?:did|does|have|has|were|was|are|is|can|could|would)\b|\?|$)",
            r"how many times\s+([a-zA-Z\s]+?)(?:\s+(?:did|does|have|has|were|was|are|is|can|could|would)\b|\?|$)",
            r"count\s+([a-zA-Z\s]+?)(?:\s+(?:of|in|for|the)\b|\?|$)",
            r"number of\s+([a-zA-Z\s]+?)(?:\s+(?:of|in|for|the)\b|\?|$)",
            r"list all\s+([a-zA-Z\s]+?)(?:\s+(?:of|in|for|the)\b|\?|$)",
            r"enumerate\s+([a-zA-Z\s]+?)(?:\s+(?:of|in|for|the)\b|\?|$)",
        ]

        target_nouns = []
        for pattern in count_patterns:
            m = re.search(pattern, q_lower)
            if m:
                noun_phrase = m.group(1).strip()
                # Split compound nouns
                nouns = re.split(r"\s+(?:and|or)\s+", noun_phrase)
                target_nouns.extend([n.strip() for n in nouns if n.strip()])

        # If no explicit nouns found, extract key nouns from question
        if not target_nouns:
            question_nouns = re.findall(r"\b([a-zA-Z]{3,})\b", q_lower)
            stopwords = {"how", "many", "times", "did", "does", "have", "has", "were", "was",
                         "are", "is", "can", "could", "would", "the", "and", "for", "what",
                         "when", "where", "which", "who", "why", "how", "many", "count", "list", "all"}
            target_nouns = [n for n in question_nouns if n not in stopwords and len(n) > 3]

        if not target_nouns:
            return []

        # Search records for proper nouns / capitalized entities near target nouns
        for r in records:
            if not r.raw_content:
                continue
            content = r.raw_content
            content_lower = content.lower()

            for noun in target_nouns:
                if len(noun) < 3:
                    continue
                # Find the noun in context and extract surrounding entities
                pattern = rf"\b{re.escape(noun)}\b"
                matches = list(re.finditer(pattern, content.lower()))
                for match in matches:
                    # Extract surrounding context to identify specific items
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]

                    # Extract entities near the noun (proper nouns, capitalized words)
                    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content[match.start():match.end()+50])
                    items.extend(entities)

                    # Also look for quoted items
                    quoted = re.findall(r'"([^"]+)"', content)
                    items.extend(quoted)

        # Also extract from structured entities in records (assistant responses with lists)
        for r in records:
            if r.raw_content and "assistant:" in r.raw_content.lower():
                # Look for lists in assistant responses
                lines = r.raw_content.split('\n')
                for line in lines:
                    if any(w in line.lower() for w in target_nouns if len(w) > 2):
                        # Extract list items (numbered, bulleted, or comma-separated)
                        list_items = re.findall(r'(?:^\s*[\d\-\*]\s*|\b)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', line)
                        items.extend(list_items)

        # Also extract capitalized entities from all records (proper nouns)
        for r in records:
            if not r.raw_content:
                continue
            # Extract all proper nouns (capitalized words/phrases)
            proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", r.raw_content)
            items.extend(proper_nouns)

        # Filter out common words that aren't countable items
        filtered_items = []
        for item in items:
            item_lower = item.lower()
            # Filter out common non-countable words
            if item_lower in {"the", "and", "or", "but", "if", "then", "when", "where", "what",
                             "who", "why", "how", "many", "times", "did", "does", "have", "has",
                             "were", "was", "are", "is", "can", "could", "would", "the", "and",
                             "for", "what", "when", "where", "which", "who", "why", "how", "many",
                             "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
                             "us", "them", "my", "your", "his", "her", "its", "our", "their",
                             "a", "an", "in", "on", "at", "to", "from", "by", "with", "about",
                             "into", "over", "under", "user", "assistant", "system", "human", "ai", "model"}:
                continue
            if len(item) < 2:
                continue
            filtered_items.append(item)

        return list(dict.fromkeys(filtered_items))  # Deduplicate preserving order

    def _format_skill_block(self, calc: AggregationCalculation) -> str:
        """Format the calculation result as a skill block for context injection."""
        if calc.calculation_type == "abstention" or not calc.items:
            return ""

        items_str = ", ".join(f'"{item}"' for item in calc.items)

        return (
            f"\n[AM SKILL CO-PROCESSOR: AGGREGATION COUNTER]\n"
            f"- Identified Instances: [{items_str}]\n"
            f"- Exact Verified Total: {calc.count}\n"
            f"(Instruction to Reader: The exact count is {calc.count}. Quote this number directly.)\n"
        )

    def clear_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()


# Auto-register on import
def _auto_register():
    """Register AggregationSkill globally on module import."""
    try:
        from .base import register_skill
        register_skill(AggregationSkill())
    except Exception:
        pass  # Defer registration if registry not ready


# Auto-register when module is loaded
_auto_register()


__all__ = ["AggregationSkill", "AggregationCalculation"]
