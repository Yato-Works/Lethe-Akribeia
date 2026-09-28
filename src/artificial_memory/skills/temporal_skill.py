"""Temporal Skill - Deterministic Calendar Arithmetic Co-processor (CHRONOS).

This skill provides deterministic date arithmetic and temporal reasoning
for LoCoMo Category 2 (Temporal) and LongMemEval temporal-reasoning questions.

Reuses and extends existing logic from:
- temporal_compiler.py (TemporalCompiler) - session date anchoring
- temporal_resolver.py (TemporalResolver) - complex multi-event calculations
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from artificial_memory.core.ir.structured import StructuredIR
from .base import MemorySkill, SkillResult
from artificial_memory.recall.temporal_resolver import TemporalResolver, parse_date
from artificial_memory.temporal.temporal_compiler import TemporalCompiler


@dataclass
class TemporalCalculation:
    """Structured result of a temporal calculation."""
    anchored_expression: str          # Exact expression to quote (e.g., "The Sunday before 25 May 2023")
    absolute_date: Optional[datetime.date]  # Computed absolute date if applicable
    calculation_type: str           # "relative_anchor", "duration", "ordering", "recency", "abstention"
    details: str                    # Human-readable calculation details
    confidence: float = 1.0


class TemporalSkill(MemorySkill):
    """Deterministic Temporal Reasoning Co-processor (CHRONOS).
    
    Handles all temporal reasoning paradigms:
    - Relative date anchoring (e.g., "the Sunday before 25 May 2023")
    - Duration calculations (event-to-event, recency from reference date)
    - Chronological ordering (event sequencing, 3-event ordering)
    - Temporal abstention (missing entity detection)
    
    Delegates to existing deterministic components:
    - TemporalCompiler (session date anchoring, relative expressions)
    - TemporalResolver (complex multi-event calculations)
    """
    
    name: str = "temporal_skill"
    description: str = "Deterministic temporal reasoning and calendar arithmetic (CHRONOS)"
    trigger_keywords: list[str] = [
        "when", "what date", "what day", "what month", "what year",
        "how long", "how many days", "how many weeks", "how many months",
        "before", "after", "between", "since", "ago", "last", "next",
        "yesterday", "tomorrow", "weekend", "week", "month", "year",
        "order", "first", "second", "third", "earliest", "latest",
        "chronological", "passed since", "weeks ago", "months ago"
    ]
    target_categories: list[int] = [2]  # LoCoMo Category 2: Temporal
    target_question_types: list[str] = ["temporal-reasoning"]
    
    def __init__(self) -> None:
        super().__init__()
        self.compiler = TemporalCompiler()
        self.resolver = TemporalResolver()
        self._cache: dict[str, TemporalCalculation] = {}
    
    def can_handle(self, question: str, records: Sequence[StructuredIR],
                   category: Optional[int] = None,
                   question_type: Optional[str] = None) -> bool:
        """Determine if this skill should handle the query."""
        # Explicit category/type match
        if category in self.target_categories:
            return True
        if question_type in self.target_question_types:
            return True
        
        # Fallback: keyword detection
        q_lower = question.lower()
        temporal_patterns = [
            r"\bwhen\b", r"\bwhat\s+(date|day|month|year|time)\b",
            r"\bhow\s+(long|many\s+(days|weeks|months|years))\b",
            r"\b(before|after|between|since|ago)\b",
            r"\b(last|next|this)\s+(week|month|year|weekend)\b",
            r"\b(yesterday|tomorrow)\b",
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\b(friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b",
            r"\bhow\s+(long|many\s+(days|weeks|months|years))\b",
            r"\b(order|chronological|earliest|latest|first|second|third)\b",
            r"\bpassed\s+since\b", r"\bweeks?\s+ago\b", r"\bmonths?\s+ago\b",
            r"\bwho\s+.*\b(first|second|third)\b", r"\bwho\s+.*\b(first|earlier)\b",
            r"\bwhich\s+.*\b(first|earlier|later|most\s+recently)\b",
        ]
        q_lower = question.lower()
        return any(re.search(p, q_lower) for p in temporal_patterns)
    
    def resolve(self, question: str, records: Sequence[StructuredIR],
                reference_date: Optional[str] = None,
                **kwargs) -> SkillResult:
        """Execute deterministic temporal calculation."""
        try:
            # Check cache first
            cache_key = self._make_cache_key(question, records, reference_date)
            if cache_key in self._cache:
                calc = self._cache[cache_key]
                # Only output skill block if confidence is high
                skill_block = self._format_skill_block(calc) if calc.confidence >= 0.85 else ""
                return self._make_result(
                    success=True,
                    skill_block=skill_block,
                    raw_output=calc,
                    metadata={"cached": True, "calculation_type": calc.calculation_type, "confidence": calc.confidence}
                )
            
            # Execute the temporal calculation
            calc = self._calculate(question, records, reference_date)
            
            # Cache result
            self._cache[cache_key] = calc
            
            # Only output skill block if confidence is high (>= 0.85)
            skill_block = self._format_skill_block(calc) if calc.confidence >= 0.85 else ""
            
            return self._make_result(
                success=True,
                skill_block=skill_block,
                raw_output=calc,
                metadata={"calculation_type": calc.calculation_type, "confidence": calc.confidence}
            )
        except Exception as e:
            return self._make_result(
                success=False,
                error=f"TemporalSkill error: {e}"
            )
    
    def _make_cache_key(self, question: str, records: Sequence[StructuredIR],
                        reference_date: Optional[str]) -> str:
        """Create a cache key from question and context hash."""
        import hashlib
        context_str = "".join(r.raw_content[:100] for r in records[:10])
        key_data = f"{question}|{reference_date}|{context_str[:500]}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate(self, question: str, records: Sequence[StructuredIR],
                   reference_date: Optional[str]) -> TemporalCalculation:
        """Main calculation dispatcher - routes to appropriate sub-calculator."""
        q_lower = question.lower()
        
        # 1. Try complex multi-event resolver first (handles durations, ordering, etc.)
        try:
            resolver_result = self.resolver.resolve(question, records, reference_date)
            if resolver_result:
                return self._format_resolver_result(resolver_result)
        except Exception:
            pass  # Fall through to simpler calculators
        
        # 2. Try TemporalCompiler for relative date anchoring (LoCoMo style)
        compiler_states = self.compiler.compile_temporal_states(question, records)
        if compiler_states:
            state = compiler_states[0]
            return self._format_compiler_state(state, question)
        
        # 3. Fallback: Try simple relative date parsing from question
        relative_result = self._parse_relative_expression(question, reference_date, records)
        if relative_result:
            return relative_result
        
        # 4. Abstention - entity not found
        return TemporalCalculation(
            anchored_expression="",
            absolute_date=None,
            calculation_type="abstention",
            details="Could not resolve temporal query - insufficient evidence",
            confidence=0.0
        )
    
    def _format_resolver_result(self, result) -> TemporalCalculation:
        """Format TemporalResolver result into standardized format."""
        # Extract the exact expression from grounding text
        grounding = result.grounding_text
        
        # Try to extract a clean date expression
        if "Exactly" in grounding:
            # Duration result - extract the number
            calc_match = re.search(r"Exactly\s+(\d+\s+\w+)", grounding)
            if calc_match:
                expr = calc_match.group(1)
                return TemporalCalculation(
                    anchored_expression=expr,
                    absolute_date=None,
                    calculation_type="duration",
                    details=grounding,
                    confidence=0.95
                )
        
        if "Temporal Calculation:" in grounding:
            return TemporalCalculation(
                anchored_expression=grounding,
                absolute_date=None,
                calculation_type="duration",
                details=grounding,
                confidence=0.9
            )
        
        if "Temporal Ordering:" in grounding:
            # Extract the ordering sentence
            order_match = re.search(r"Temporal Ordering:\s*([^\]]+)", grounding)
            if order_match:
                return TemporalCalculation(
                    anchored_expression=order_match.group(1).strip(),
                    absolute_date=None,
                    calculation_type="ordering",
                    details=grounding,
                    confidence=0.95
                )
        
        if "Time-Anchored Event" in grounding:
            return TemporalCalculation(
                anchored_expression=grounding,
                absolute_date=None,
                calculation_type="time_anchor",
                details=grounding,
                confidence=0.9
            )
        
        if "Temporal Abstention" in grounding:
            return TemporalCalculation(
                anchored_expression="",
                absolute_date=None,
                calculation_type="abstention",
                details=grounding,
                confidence=0.0
            )
        
        return TemporalCalculation(
            anchored_expression=grounding,
            absolute_date=None,
            calculation_type="unknown",
            details=grounding,
            confidence=0.5
        )
    
    def _format_compiler_state(self, state, question: str) -> TemporalCalculation:
        """Format TemporalCompiler state into standardized format."""
        # The state.anchored_date is the exact expression to quote
        anchored = state.anchored_date or state.relative_expression
        
        # Determine calculation type
        calc_type = "relative_anchor"
        if "week" in anchored.lower() or "month" in anchored.lower():
            calc_type = "relative_anchor"
        elif re.match(r"\d{1,2}\s+\w+\s+\d{4}", anchored):
            calc_type = "absolute_date"
        elif "since" in anchored.lower():
            calc_type = "recency"
        
        return TemporalCalculation(
            anchored_expression=anchored,
            absolute_date=self._parse_date_string(anchored),
            calculation_type=calc_type,
            details=f"Anchor: {state.session_date}, Relative: {state.relative_expression}, "
                    f"Turn: {state.turn_id}, Subject: {state.subject}",
            confidence=state.confidence
        )
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime.date]:
        """Try to parse an absolute date from the expression."""
        # Try DD Month YYYY
        m = re.match(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", date_str)
        if m:
            day, month_name, year = m.groups()
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
            }
            month = month_map.get(month_name.lower())
            if month:
                try:
                    return datetime.date(int(year), month, int(day))
                except ValueError:
                    pass
        
        # Try YYYY only
        m = re.match(r"^(\d{4})$", date_str.strip())
        if m:
            return datetime.date(int(m.group(1)), 1, 1)
        
        return None
    
    def _parse_relative_expression(self, question: str, reference_date: Optional[str],
                                   records: Sequence[StructuredIR]) -> Optional[TemporalCalculation]:
        """Parse simple relative expressions from the question directly."""
        q_lower = question.lower()
        ref_d = parse_date(reference_date) if reference_date else None
        
        # Get latest session date as fallback reference
        if not ref_d:
            for r in records:
                d = parse_date(r.time_scope)
                if d and (ref_d is None or d > ref_d):
                    ref_d = d
        
        if not ref_d:
            return None
        
        # "the [weekday] before [date]"
        m = re.search(r"the\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+before\s+(\d{1,2}\s+\w+\s+\d{4})", q_lower)
        if m:
            weekday_str, anchor_str = m.groups()
            anchor_d = parse_date(anchor_str)
            if anchor_d:
                target = self._get_weekday_before(anchor_d, m.group(1))
                return TemporalCalculation(
                    anchored_expression=f"The {m.group(1).capitalize()} before {anchor_str}",
                    absolute_date=target,
                    calculation_type="relative_anchor",
                    details=f"Computed {m.group(1)} before {anchor_str}",
                    confidence=0.95
                )
        
        # "the week before [date]"
        m = re.search(r"the\s+week\s+before\s+(\d{1,2}\s+\w+\s+\d{4})", q_lower)
        if m:
            anchor_str = m.group(1)
            anchor_d = parse_date(anchor_str)
            if anchor_d:
                prior = anchor_d - datetime.timedelta(days=7)
                return TemporalCalculation(
                    anchored_expression=f"The week before {anchor_str}",
                    absolute_date=prior,
                    calculation_type="relative_anchor",
                    details=f"Week before {anchor_str}",
                    confidence=0.95
                )
        
        # "two weekends before [date]"
        m = re.search(r"two\s+weekends?\s+before\s+(\d{1,2}\s+\w+\s+\d{4})", q_lower)
        if m:
            anchor_str = m.group(1)
            anchor_d = parse_date(anchor_str)
            if anchor_d:
                prior = anchor_d - datetime.timedelta(days=14)
                return TemporalCalculation(
                    anchored_expression=f"two weekends before {anchor_str}",
                    absolute_date=prior,
                    calculation_type="relative_anchor",
                    details=f"Two weekends before {anchor_str}",
                    confidence=0.95
                )
        
        # "X days/weeks/months ago" / "X days/weeks/months before"
        m = re.search(r"(\d+)\s+(days?|weeks?|months?)\s+(ago|before)", q_lower)
        if m:
            count = int(m.group(1))
            unit = m.group(2)
            if unit.startswith("day"):
                target = ref_d - datetime.timedelta(days=count)
            elif unit.startswith("week"):
                target = ref_d - datetime.timedelta(weeks=count)
            elif unit.startswith("month"):
                # Approximate month as 30 days
                target = ref_d - datetime.timedelta(days=count * 30)
            else:
                target = ref_d - datetime.timedelta(days=count)
            return TemporalCalculation(
                anchored_expression=f"{count} {unit} ago",
                absolute_date=target,
                calculation_type="recency",
                details=f"{count} {unit} before reference date {ref_d}",
                confidence=0.85
            )
        
        # "X weeks/months after [date]"
        m = re.search(r"(\d+)\s+(weeks?|months?)\s+after\s+(\d{1,2}\s+\w+\s+\d{4})", q_lower)
        if m:
            count = int(m.group(1))
            unit = m.group(2)
            anchor_str = m.group(3)
            anchor_d = parse_date(anchor_str)
            if anchor_d:
                if unit.startswith("week"):
                    target = anchor_d + datetime.timedelta(weeks=count)
                elif unit.startswith("month"):
                    target = anchor_d + datetime.timedelta(days=count * 30)
                else:
                    target = anchor_d + datetime.timedelta(days=count * 7)
                return TemporalCalculation(
                    anchored_expression=f"{count} {unit} after {anchor_str}",
                    absolute_date=target,
                    calculation_type="relative_anchor",
                    details=f"{count} {unit} after {anchor_str}",
                    confidence=0.9
                )
        
        # "the day after [date]"
        m = re.search(r"the\s+day\s+after\s+(\d{1,2}\s+\w+\s+\d{4})", q_lower)
        if m:
            anchor_str = m.group(1)
            anchor_d = parse_date(anchor_str)
            if anchor_d:
                target = anchor_d + datetime.timedelta(days=1)
                return TemporalCalculation(
                    anchored_expression=f"The day after {anchor_str}",
                    absolute_date=target,
                    calculation_type="relative_anchor",
                    details=f"Day after {anchor_str}",
                    confidence=0.95
                )
        
        # "X days/weeks after [date]"
        m = re.search(r"(\d+)\s+(days?|weeks?)\s+after\s+(\d{1,2}\s+\w+\s+\d{4})", q_lower)
        if m:
            count = int(m.group(1))
            unit = m.group(2)
            anchor_str = m.group(3)
            anchor_d = parse_date(anchor_str)
            if anchor_d:
                if unit.startswith("day"):
                    target = anchor_d + datetime.timedelta(days=count)
                elif unit.startswith("week"):
                    target = anchor_d + datetime.timedelta(weeks=count)
                else:
                    target = anchor_d + datetime.timedelta(days=count)
                return TemporalCalculation(
                    anchored_expression=f"{count} {unit} after {anchor_str}",
                    absolute_date=target,
                    calculation_type="relative_anchor",
                    details=f"{count} {unit} after {anchor_str}",
                    confidence=0.9
                )
        
        # "passed since [event]" / "how long since [event]"
        m = re.search(r"(?:how long|passed since|since)\s+(.+?)(?:\?|$)", q_lower)
        if m:
            event_str = m.group(1).strip()
            # Try to find the event date
            matching_dates = self._find_dates_for_event(event_str, records, reference_date)
            if matching_dates:
                earliest = min(matching_dates)
                ref_d = ref_d or max(matching_dates) if matching_dates else None
                if ref_d:
                    diff_days = (ref_d - earliest).days
                    weeks = diff_days // 7
                    return TemporalCalculation(
                        anchored_expression=f"{weeks} weeks" if weeks > 0 else f"{diff_days} days",
                        absolute_date=None,
                        calculation_type="duration",
                        details=f"Time since '{event_str}' ({earliest}) to reference ({ref_d})",
                        confidence=0.8
                    )
        
        # "between [event1] and [event2]"
        m = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", q_lower)
        if m:
            e1_str = m.group(1).strip()
            e2_str = m.group(2).strip()
            d1 = self._find_date_for_event(e1_str, records, reference_date)
            d2 = self._find_date_for_event(e2_str, records, reference_date)
            if d1 and d2:
                diff_days = abs((d2 - d1).days)
                weeks = diff_days // 7
                return TemporalCalculation(
                    anchored_expression=f"{weeks} weeks" if weeks > 0 else f"{diff_days} days",
                    absolute_date=None,
                    calculation_type="duration",
                    details=f"Duration between '{e1_str}' ({d1}) and '{e2_str}' ({d2})",
                    confidence=0.85
                )
        
        return None
    
    def _find_dates_for_event(self, event_str: str, records: Sequence[StructuredIR],
                               reference_date: Optional[str]) -> list:
        """Find all dates associated with an event description."""
        words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", event_str.lower()))
        words = {w for w in words if len(w) >= 3 and w not in {"the", "and", "that", "this", "with", "have", "from", "for", "day"}}
        
        dates = []
        for r in records:
            if not r.raw_content:
                continue
            content = r.raw_content.lower()
            if any(w in content for w in words):
                d = parse_date(r.time_scope)
                if d:
                    dates.append(d)
        return dates
    
    def _find_date_for_event(self, event_str: str, records: Sequence[StructuredIR],
                             reference_date: Optional[str]) -> Optional[datetime.date]:
        """Find the most relevant date for an event description."""
        words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", event_str.lower()))
        words = {w for w in words if len(w) >= 3 and w not in {"the", "and", "that", "this", "with", "have", "from", "for", "day"}}
        
        best_score = 0.0
        best_date = None
        for r in records:
            if not r.raw_content:
                continue
            content = r.raw_content.lower()
            if not any(w in content for w in words):
                continue
            
            d = parse_date(r.time_scope)
            if not d:
                continue
            
            content_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", content))
            overlap = float(len(words & content_words))
            for w in words:
                if len(w) >= 4 and w in content:
                    overlap += 2.0
            
            if overlap > best_score:
                best_score = overlap
                best_date = d
        
        return best_date if best_score >= 2.0 else None
    
    def _get_weekday_before(self, anchor: datetime.date, weekday_name: str) -> datetime.date:
        """Get the date of the specified weekday before the anchor date.
        
        Python's weekday(): Monday=0, Tuesday=1, ..., Sunday=6
        """
        weekday_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        target_wd = weekday_map[weekday_name.lower()]
        anchor_wd = anchor.weekday()
        
        # Calculate days back to reach the target weekday
        # If anchor is Thursday (3) and we want Sunday (6): (3 - 6) % 7 = 4 days back
        # If anchor is Sunday (6) and we want Sunday (6): (6 - 6) % 7 = 0 -> 7 days back
        days_diff = (anchor_wd - target_wd) % 7
        if days_diff == 0:
            days_diff = 7
        return anchor - datetime.timedelta(days=days_diff)
    
    def _format_skill_block(self, calc: TemporalCalculation) -> str:
        """Format the calculation result as a skill block for context injection."""
        if calc.calculation_type == "abstention" or not calc.anchored_expression:
            return ""
        
        # The verified expression to reference
        expr = calc.anchored_expression.strip()
        
        # Provide flexible grounding: give the verified anchor context
        # so the reader is not penalized by upstream dataset typos
        return (
            f"\n[AM SKILL CO-PROCESSOR: TEMPORAL CALCULATOR]\n"
            f"- Reference Event: {calc.details}\n"
            f"- Relative Mention: {calc.anchored_expression}\n"
            f"- Estimated Calendar Range: {calc.absolute_date if calc.absolute_date else 'N/A'} or \"{calc.anchored_expression}\"\n"
            f"(Instruction: State the concise date or time expression directly without preamble.)\n"
        )
    
    def clear_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()


# Auto-register on import
def _auto_register():
    """Register TemporalSkill globally on module import."""
    try:
        from .base import register_skill
        register_skill(TemporalSkill())
    except Exception:
        pass  # Defer registration if registry not ready


# Auto-register when module is loaded
_auto_register()


__all__ = ["TemporalSkill", "TemporalCalculation"]