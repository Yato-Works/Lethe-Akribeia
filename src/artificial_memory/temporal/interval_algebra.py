"""Allen's Interval Algebra for Deterministic Temporal Reasoning (CHRONOS III).

Implements the 13 fundamental relations of Allen's Interval Algebra (1983)
for reasoning over time intervals and discrete time points in long-term AI memory.

Relations:
- BEFORE (p < q), AFTER (p > q)
- MEETS (p.end == q.start), MET_BY (p.start == q.end)
- OVERLAPS, OVERLAPPED_BY
- DURING, CONTAINS
- STARTS, STARTED_BY
- FINISHES, FINISHED_BY
- EQUALS
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from artificial_memory.recall.temporal_resolver import to_comparable_datetime


class IntervalRelation(StrEnum):
    """The 13 primitive relations between two time intervals."""
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    DURING = "during"
    CONTAINS = "contains"
    STARTS = "starts"
    STARTED_BY = "started_by"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


@dataclass(frozen=True)
class TimeInterval:
    """A bounded temporal interval [start, end].

    If start == end, represents a discrete time point (instant).
    """
    start: datetime.datetime | datetime.date
    end: datetime.datetime | datetime.date
    label: str = ""

    def __post_init__(self) -> None:
        dt_start = to_comparable_datetime(self.start)
        dt_end = to_comparable_datetime(self.end)
        if dt_start > dt_end:
            raise ValueError(f"Interval start ({self.start}) cannot be after end ({self.end})")

    @property
    def is_point(self) -> bool:
        """True if the interval represents a single point in time."""
        return to_comparable_datetime(self.start) == to_comparable_datetime(self.end)

    @property
    def duration_days(self) -> int:
        """Duration of the interval in whole days."""
        dt_start = to_comparable_datetime(self.start)
        dt_end = to_comparable_datetime(self.end)
        return (dt_end.date() - dt_start.date()).days

    @classmethod
    def point(cls, at: datetime.datetime | datetime.date, label: str = "") -> Self:
        """Create a point interval where start == end."""
        return cls(start=at, end=at, label=label)

    def relation_to(self, other: TimeInterval) -> IntervalRelation:
        """Determine Allen's interval relation from self to other."""
        s1 = to_comparable_datetime(self.start)
        e1 = to_comparable_datetime(self.end)
        s2 = to_comparable_datetime(other.start)
        e2 = to_comparable_datetime(other.end)

        if e1 < s2:
            return IntervalRelation.BEFORE
        if s1 > e2:
            return IntervalRelation.AFTER
        if e1 == s2:
            return IntervalRelation.MEETS
        if s1 == e2:
            return IntervalRelation.MET_BY
        if s1 == s2 and e1 == e2:
            return IntervalRelation.EQUALS
        if s1 == s2:
            return IntervalRelation.STARTS if e1 < e2 else IntervalRelation.STARTED_BY
        if e1 == e2:
            return IntervalRelation.FINISHES if s1 > s2 else IntervalRelation.FINISHED_BY
        if s1 > s2 and e1 < e2:
            return IntervalRelation.DURING
        if s1 < s2 and e1 > e2:
            return IntervalRelation.CONTAINS
        if s1 < s2 and s2 < e1 < e2:
            return IntervalRelation.OVERLAPS
        if s2 < s1 and s1 < e2 < e1:
            return IntervalRelation.OVERLAPPED_BY

        # Fallback for point containment edge cases
        if s1 >= s2 and e1 <= e2:
            return IntervalRelation.DURING
        return IntervalRelation.CONTAINS

    def overlaps_with(self, other: TimeInterval) -> bool:
        """Check if self and other share any non-empty duration."""
        rel = self.relation_to(other)
        return rel in {
            IntervalRelation.OVERLAPS,
            IntervalRelation.OVERLAPPED_BY,
            IntervalRelation.DURING,
            IntervalRelation.CONTAINS,
            IntervalRelation.STARTS,
            IntervalRelation.STARTED_BY,
            IntervalRelation.FINISHES,
            IntervalRelation.FINISHED_BY,
            IntervalRelation.EQUALS,
        }

    def format_derivation(self, other: TimeInterval) -> str:
        """Human-readable deterministic deduction of the relationship between two events."""
        rel = self.relation_to(other)
        lbl1 = self.label or f"[{self.start}..{self.end}]"
        lbl2 = other.label or f"[{other.start}..{other.end}]"

        explanations = {
            IntervalRelation.BEFORE: f"'{lbl1}' completed before '{lbl2}' began.",
            IntervalRelation.AFTER: f"'{lbl1}' began after '{lbl2}' concluded.",
            IntervalRelation.MEETS: f"'{lbl1}' concluded exactly as '{lbl2}' started.",
            IntervalRelation.MET_BY: f"'{lbl1}' started immediately when '{lbl2}' ended.",
            IntervalRelation.OVERLAPS: f"'{lbl1}' started earlier and overlapped with '{lbl2}'.",
            IntervalRelation.OVERLAPPED_BY: f"'{lbl1}' started during '{lbl2}' and extended beyond it.",
            IntervalRelation.DURING: f"'{lbl1}' took place entirely within the timeframe of '{lbl2}'.",
            IntervalRelation.CONTAINS: f"'{lbl1}' encompassed the entire duration of '{lbl2}'.",
            IntervalRelation.STARTS: f"'{lbl1}' started simultaneously with '{lbl2}' but finished earlier.",
            IntervalRelation.STARTED_BY: f"'{lbl1}' started simultaneously with '{lbl2}' and lasted longer.",
            IntervalRelation.FINISHES: f"'{lbl1}' started later than '{lbl2}' and finished simultaneously.",
            IntervalRelation.FINISHED_BY: f"'{lbl1}' started earlier than '{lbl2}' and finished simultaneously.",
            IntervalRelation.EQUALS: f"'{lbl1}' and '{lbl2}' covered the exact same timeframe.",
        }
        return f"[ALLEN INTERVAL DEDUCTION]: {explanations[rel]}"


# ---------------------------------------------------------------------------
# Temporal Algebraic AST (Abstract Syntax Tree) for Compound Reasoning (P3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TemporalASTNode:
    """Base class for temporal algebraic expressions."""


@dataclass(frozen=True)
class DatePointNode(TemporalASTNode):
    """A concrete calendar date."""
    date: datetime.date


@dataclass(frozen=True)
class OffsetNode(TemporalASTNode):
    """An offset relative to a base temporal node.

    direction: +1 for after/later/from, -1 for before/earlier/prior.
    """
    base: TemporalASTNode
    amount: int
    unit: str  # 'days', 'weeks', 'months', 'years'
    direction: int = 1


@dataclass(frozen=True)
class OrdinalWeekdayNode(TemporalASTNode):
    """N-th weekday of a month/year (e.g. 3rd Monday of June 2023).

    ordinal: 1..5, or -1 for last.
    weekday: 0=Monday .. 6=Sunday.
    """
    ordinal: int
    weekday: int
    month: int
    year: int


@dataclass(frozen=True)
class IntervalSpanNode(TemporalASTNode):
    """An interval bounded by start and end nodes."""
    start: TemporalASTNode
    end: TemporalASTNode
    label: str = ""


def resolve_ast(node: TemporalASTNode) -> datetime.date | TimeInterval:
    """Deterministically evaluate a temporal AST node to a concrete date or interval."""
    import calendar

    if isinstance(node, DatePointNode):
        return node.date

    if isinstance(node, OffsetNode):
        base_val = resolve_ast(node.base)
        if isinstance(base_val, TimeInterval):
            base_date = base_val.end if node.direction > 0 else base_val.start
            base_date = base_date.date() if isinstance(base_date, datetime.datetime) else base_date
        elif isinstance(base_val, datetime.datetime):
            base_date = base_val.date()
        else:
            base_date = base_val

        delta = node.direction * node.amount
        unit = node.unit.lower()
        if unit.startswith("day"):
            return base_date + datetime.timedelta(days=delta)
        elif unit.startswith("week"):
            return base_date + datetime.timedelta(days=delta * 7)
        elif unit.startswith("month"):
            new_month = base_date.month + delta
            new_year = base_date.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            max_day = calendar.monthrange(new_year, new_month)[1]
            return datetime.date(new_year, new_month, min(base_date.day, max_day))
        elif unit.startswith("year"):
            new_year = base_date.year + delta
            max_day = calendar.monthrange(new_year, base_date.month)[1]
            return datetime.date(new_year, base_date.month, min(base_date.day, max_day))
        raise ValueError(f"Unknown temporal unit: {node.unit}")

    if isinstance(node, OrdinalWeekdayNode):
        import calendar
        c = calendar.Calendar(firstweekday=0)
        matching = [
            d for d in c.itermonthdates(node.year, node.month)
            if d.month == node.month and d.weekday() == node.weekday
        ]
        if not matching:
            raise ValueError(f"No weekday {node.weekday} found in {node.year}-{node.month}")
        if node.ordinal == -1 or node.ordinal > len(matching):
            return matching[-1]
        return matching[node.ordinal - 1]

    if isinstance(node, IntervalSpanNode):
        s = resolve_ast(node.start)
        e = resolve_ast(node.end)
        s_date = s.start if isinstance(s, TimeInterval) else s
        e_date = e.end if isinstance(e, TimeInterval) else e
        return TimeInterval(start=s_date, end=e_date, label=node.label)

    raise TypeError(f"Unsupported AST node type: {type(node)}")


_WORD_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "a couple of": 2, "a couple": 2, "a few": 3,
}
_ORDINALS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1,
}
_WEEKDAYS: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_compound_temporal_expression(
    expr: str,
    ref_date: datetime.date | None = None,
) -> TemporalASTNode | None:
    """Parse complex compound temporal expressions into an algebraic AST.

    Supports:
    - Offset expressions: "2 weeks after 25 December 2022", "3 days before 15 July 2023",
      "two months after last Christmas"
    - Ordinal weekday expressions: "the third Monday of June 2023", "the last Friday of next month"
    - Interval span expressions: "from May 2021 until 4 months later"
    """
    import re

    clean = expr.strip().lower()
    ref = ref_date or datetime.date.today()

    # Helper: resolve simple anchor expressions (e.g. "christmas", "25 december 2022")
    def resolve_anchor(text: str) -> datetime.date | None:
        t = text.strip().lower()
        if "christmas" in t:
            # Christmas in reference year or previous year
            yr = ref.year if "last" not in t else (ref.year - 1 if ref.month < 12 else ref.year)
            return datetime.date(yr, 12, 25)
        if "new year" in t:
            yr = ref.year if "last" not in t else ref.year - 1
            return datetime.date(yr, 1, 1)

        # "DD Month YYYY" or "DD Month, YYYY"
        m_dmy = re.search(r"\b(\d{1,2})\s+([a-z]+),?\s+(\d{4})\b", t)
        if m_dmy:
            day = int(m_dmy.group(1))
            m_str = m_dmy.group(2)
            m_num = _MONTHS.get(m_str) or _MONTHS.get(m_str[:3])
            year = int(m_dmy.group(3))
            if m_num:
                return datetime.date(year, m_num, day)

        # "Month YYYY"
        m_my = re.search(r"\b([a-z]+)\s+(\d{4})\b", t)
        if m_my:
            m_str = m_my.group(1)
            m_num = _MONTHS.get(m_str) or _MONTHS.get(m_str[:3])
            year = int(m_my.group(2))
            if m_num:
                return datetime.date(year, m_num, 1)

        # Bare ISO YYYY-MM-DD
        m_iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", t)
        if m_iso:
            return datetime.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))

        return None

    # Pattern 1: Interval "from <A> until/to <B>"
    m_span = re.match(r"(?:from\s+)?(.+?)\s+(?:until|to)\s+(.+)", clean)
    if m_span and ("from" in clean or "until" in clean):
        start_str = m_span.group(1)
        end_str = m_span.group(2)
        start_node = parse_compound_temporal_expression(start_str, ref)
        if not start_node:
            d_start = resolve_anchor(start_str)
            if d_start:
                start_node = DatePointNode(d_start)
        if start_node:
            end_node = parse_compound_temporal_expression(end_str, ref)
            if not end_node:
                d_end = resolve_anchor(end_str)
                if d_end:
                    end_node = DatePointNode(d_end)
            if end_node:
                return IntervalSpanNode(start=start_node, end=end_node)

    # Pattern 2: Offset expressions ("2 weeks after 25 December 2022")
    m_offset = re.match(
        r"^(?:the\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a\s+few|a\s+couple(?:\s+of)?)\s+"
        r"(day|week|month|year)s?\s+(after|before|later|earlier|from)\s+(.+)$",
        clean,
    )
    if m_offset:
        amt_str = m_offset.group(1)
        unit = m_offset.group(2)
        direction_word = m_offset.group(3)
        anchor_str = m_offset.group(4)

        amount = int(amt_str) if amt_str.isdigit() else _WORD_NUMBERS.get(amt_str, 1)
        direction = -1 if direction_word in ("before", "earlier") else 1

        anchor_node = parse_compound_temporal_expression(anchor_str, ref)
        if not anchor_node:
            d_anchor = resolve_anchor(anchor_str)
            if d_anchor:
                anchor_node = DatePointNode(d_anchor)

        if anchor_node:
            return OffsetNode(base=anchor_node, amount=amount, unit=unit, direction=direction)

    # Pattern 3: Ordinal weekday expressions ("the third Monday of June 2023", "the last Friday of next month")
    m_ordinal = re.match(
        r"^(?:the\s+)?(first|second|third|fourth|fifth|last)\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+of\s+(.+)$",
        clean,
    )
    if m_ordinal:
        ord_str = m_ordinal.group(1)
        wday_str = m_ordinal.group(2)
        month_yr_str = m_ordinal.group(3).strip()

        ordinal = _ORDINALS.get(ord_str, 1)
        weekday = _WEEKDAYS.get(wday_str, 0)

        # Resolve month and year
        if "next month" in month_yr_str:
            nm = ref.month + 1
            ny = ref.year + (1 if nm > 12 else 0)
            nm = 1 if nm > 12 else nm
            return OrdinalWeekdayNode(ordinal=ordinal, weekday=weekday, month=nm, year=ny)
        elif "this month" in month_yr_str:
            return OrdinalWeekdayNode(ordinal=ordinal, weekday=weekday, month=ref.month, year=ref.year)
        else:
            d = resolve_anchor(month_yr_str)
            if d:
                return OrdinalWeekdayNode(ordinal=ordinal, weekday=weekday, month=d.month, year=d.year)

    # Fallback to single concrete date
    d_single = resolve_anchor(clean)
    if d_single:
        return DatePointNode(d_single)

    return None


# ─── Canonical Temporal Form for Official F1 ───
_CANONICAL_MONTHS = {
    1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
    7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december",
}


def canonicalize_temporal_answer(date_obj: datetime.date | datetime.datetime | str) -> str:
    """Convert any temporal representation to official-scorer-friendly canonical form."""
    if isinstance(date_obj, str):
        for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%B %Y", "%Y"):
            try:
                date_obj = datetime.datetime.strptime(date_obj, fmt).date()
                break
            except ValueError:
                continue
        else:
            return date_obj

    if isinstance(date_obj, datetime.datetime):
        date_obj = date_obj.date()

    if isinstance(date_obj, datetime.date):
        return f"{date_obj.day} {_CANONICAL_MONTHS[date_obj.month]} {date_obj.year}"

    return str(date_obj)


def normalize_temporal_for_scoring(answer: str, ref_date: datetime.date | None = None) -> str:
    """Post-process reader's temporal answer to canonical form before official scoring."""
    ans_clean = answer.strip()
    m_month_yr = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b",
        ans_clean.lower(),
    )
    if m_month_yr and len(ans_clean.split()) <= 4:
        return f"{m_month_yr.group(1)} {m_month_yr.group(2)}"

    parsed = parse_compound_temporal_expression(ans_clean, ref_date=ref_date)
    if parsed:
        try:
            resolved = resolve_ast(parsed)
            if isinstance(resolved, datetime.date):
                return canonicalize_temporal_answer(resolved)
            if isinstance(resolved, TimeInterval):
                return canonicalize_temporal_answer(resolved.start)
        except Exception:
            pass
    return ans_clean

