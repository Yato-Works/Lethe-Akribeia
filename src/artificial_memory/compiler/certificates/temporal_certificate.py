"""
Temporal Certificate Engine: Deterministic Calendar Arithmetic for Lethe-Akribeia V4.

Handles:
  1. Recency Diff: "How many days/weeks/months ago did [Event] happen?"
  2. Interval Diff: "How many days/weeks passed between [Event A] and [Event B]?"
  3. Chronological Order: "Which did I purchase/join/visit first, A or B?"
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.core.ir.structured import StructuredIR


@dataclass(frozen=True)
class TemporalCertificate:
    """Mathematical and chronological proof of temporal relationship."""
    has_certificate: bool
    query_type: str                 # "recency", "interval", "ordering", "date_anchor", "compound", "none"
    event_date_str: str | None
    reference_date_str: str | None
    difference_days: int | None
    difference_weeks: int | None
    difference_months: int | None
    ordered_events: list[tuple[str, str]] | None  # [(date, description), ...]
    formatted_certificate: str
    direct_answer_candidate: str | None
    interval_relation: str | None = None
    allen_derivation: str | None = None


def parse_iso_date(date_str: str) -> datetime.datetime | None:
    """Parse YYYY/MM/DD [HH:MM] or YYYY-MM-DD [HH:MM] into datetime.datetime."""
    if not date_str:
        return None
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[^\d\n]+(\d{1,2}):(\d{2}))?\b", date_str)
    if m:
        try:
            y = int(m.group(1))
            mo = int(m.group(2))
            d = int(m.group(3))
            hh = int(m.group(4)) if m.group(4) is not None else 0
            mm = int(m.group(5)) if m.group(5) is not None else 0
            return datetime.datetime(y, mo, d, hh, mm)
        except ValueError:
            return None
    return None


class TemporalCertificateEngine:
    """Engine that deterministically computes calendar diffs and generates certificates."""

    _RE_AGO = re.compile(r"\bhow many (days|weeks|months|years) ago did (i|we|my)\b", re.I)
    _RE_SINCE = re.compile(r"\bhow many (days|weeks|months|years) (have passed|passed) since\b", re.I)
    _RE_BETWEEN = re.compile(r"\bhow many (days|weeks|months|years) passed between\b", re.I)
    _RE_ORDER_A_OR_B = re.compile(r"\b(first|most recently|last|earlier|later)\b.*?(?:,\s*|\s+between\s+)(.+?)\s+or\s+(.+?)\??$", re.I)
    _RE_WHICH_FIRST = re.compile(
        r"\b(?:which|who)(?!\s+(?:year|month|date|day))\b.*?\b(?:first|most recently|last|earlier|later)\b(?!\s+(?:three|two|four|five|six|seven|eight|nine|ten|\d+|several|few|time|dog|cat|pet|book|movie|event|trip|year|month|week|day))",
        re.I
    )
    _RE_FIRST_LAST_3 = re.compile(r"\border from first to last:\s*(.+?),\s*(.+?),\s*(?:and\s*)?(.+?)\??$", re.I)
    _RE_RELATIVE_ANCHOR = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?|years?)\s+ago\b|\b(last (?:friday|monday|tuesday|wednesday|thursday|saturday|sunday)|the past weekend)\b", re.I)

    WORD_TO_NUM = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    def generate(self, question: str, records: Sequence[StructuredIR], reference_date_str: str | None) -> TemporalCertificate:
        """Deterministically resolve temporal question into a certificate."""
        q_lower = question.lower()
        ref_date = parse_iso_date(reference_date_str) if reference_date_str else None

        # 1. Check Recency Diff: "How many [unit] ago did..."
        m_ago = self._RE_AGO.search(q_lower) or self._RE_SINCE.search(q_lower)
        if m_ago and ref_date:
            unit = m_ago.group(1).lower()
            return self._resolve_recency(question, records, ref_date, reference_date_str, unit)

        # 2. Check 3-event sequence: "Which three events happened in the order from first to last: A, B, and C?"
        m_3 = self._RE_FIRST_LAST_3.search(question)
        if m_3:
            return self._resolve_three_event_ordering(m_3.group(1).strip(), m_3.group(2).strip(), m_3.group(3).strip(), records)

        # 3. Check Pairwise Ordering: "Which ... first, A or B?"
        m_ab = self._RE_ORDER_A_OR_B.search(question)
        if m_ab:
            target_direction = m_ab.group(1).lower()
            opt_a = m_ab.group(2).strip()
            opt_b = m_ab.group(3).strip()
            return self._resolve_pairwise_ordering(question, opt_a, opt_b, target_direction, records)

        # 4. General Which First / Most Recently
        if self._RE_WHICH_FIRST.search(q_lower):
            order_cert = self._resolve_ordering(question, records)
            if order_cert.has_certificate:
                return order_cert

        # 5. Check Between: "How many days passed between A and B"
        m_between = self._RE_BETWEEN.search(q_lower)
        if m_between:
            unit = m_between.group(1).lower()
            return self._resolve_interval(question, records, unit)

        # 6. Check Relative Anchor (e.g. "I mentioned X 4 weeks ago")
        m_rel = self._RE_RELATIVE_ANCHOR.search(q_lower)
        if m_rel and ref_date:
            anchor_cert = self._resolve_relative_anchor(question, records, ref_date, m_rel)
            if anchor_cert.has_certificate:
                return anchor_cert

        # 7. Check Compound Temporal Query via Allen's Interval Algebra
        if self._RE_WHEN_QUERY.search(q_lower):
            compound_cert = self._resolve_compound_query(question, records, ref_date, reference_date_str)
            if compound_cert.has_certificate:
                return compound_cert

        return TemporalCertificate(
            has_certificate=False,
            query_type="none",
            event_date_str=None,
            reference_date_str=reference_date_str,
            difference_days=None,
            difference_weeks=None,
            difference_months=None,
            ordered_events=None,
            formatted_certificate="",
            direct_answer_candidate=None,
        )

    _RE_REL_AGO = re.compile(
        r"\b(?:for (?:about )?)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+(days?|weeks?|months?|years?)(?:\s+ago|\s+now)?\b",
        re.I
    )
    _RE_MONTH_DAY = re.compile(
        r"\b(?:on\s+)?(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        re.I
    )
    _MONTH_MAP = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
        "november": 11, "nov": 11, "december": 12, "dec": 12
    }

    def _extract_true_event_date(self, session_date_str: str | None, text: str) -> datetime.datetime | None:
        if not session_date_str:
            return None
        base_d = parse_iso_date(session_date_str)
        if not base_d:
            return None

        # 1. Explicit month + day in text (e.g. "on January 15th")
        m_abs = self._RE_MONTH_DAY.search(text)
        if m_abs:
            m_name = m_abs.group(1).lower()
            day = int(m_abs.group(2))
            month = self._MONTH_MAP.get(m_name)
            if month:
                year = base_d.year
                if month > base_d.month + 1:
                    year -= 1
                try:
                    return datetime.datetime(year, month, day, base_d.hour, base_d.minute)
                except ValueError:
                    pass

        # 2. Relative "X ago"
        m_rel = self._RE_REL_AGO.search(text)
        if m_rel:
            num_str = m_rel.group(1).lower()
            unit = m_rel.group(2).lower()
            val = int(num_str) if num_str.isdigit() else self.WORD_TO_NUM.get(num_str, 1)

            if "day" in unit:
                return base_d - datetime.timedelta(days=val)
            elif "week" in unit:
                return base_d - datetime.timedelta(days=val * 7)
            elif "month" in unit:
                return base_d - datetime.timedelta(days=val * 30)
            elif "year" in unit:
                return base_d - datetime.timedelta(days=val * 365)

        t_low = text.lower()
        if "yesterday" in t_low:
            return base_d - datetime.timedelta(days=1)
        if "last weekend" in t_low or "the past weekend" in t_low:
            return base_d - datetime.timedelta(days=5)

        return base_d

    def _resolve_recency(self, question: str, records: Sequence[StructuredIR], ref_date: datetime.date, ref_str: str, unit: str) -> TemporalCertificate:
        """Find the target event date matching the query and compute diff from reference date."""
        q_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", question.lower())) - {
            "how", "many", "days", "weeks", "months", "years", "ago", "did", "i", "we", "my",
            "have", "passed", "since", "the", "a", "an", "on", "at", "to", "in"
        }

        best_date = None
        best_score = 0
        best_content = ""

        d_ref = ref_date.date() if isinstance(ref_date, datetime.datetime) else ref_date

        for r in records:
            r_date = self._extract_true_event_date(r.time_scope or r.raw_content, r.raw_content)
            if not r_date:
                continue
            d_r = r_date.date() if isinstance(r_date, datetime.datetime) else r_date
            if d_r >= d_ref:
                continue

            r_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", r.raw_content.lower()))
            overlap = len(q_words & r_words)
            if overlap > best_score:
                best_score = overlap
                best_date = r_date
                best_content = r.raw_content

        if not best_date or best_score == 0:
            return TemporalCertificate(
                has_certificate=False,
                query_type="recency",
                event_date_str=None,
                reference_date_str=ref_str,
                difference_days=None,
                difference_weeks=None,
                difference_months=None,
                ordered_events=None,
                formatted_certificate="",
                direct_answer_candidate=None,
            )

        d_best = best_date.date() if isinstance(best_date, datetime.datetime) else best_date
        diff_days = (d_ref - d_best).days
        diff_weeks = round(diff_days / 7)
        # Approximate calendar months
        diff_months = (d_ref.year - d_best.year) * 12 + (d_ref.month - d_best.month)

        if unit == "days":
            direct_ans = f"{diff_days} days ago"
        elif unit == "weeks":
            direct_ans = f"{diff_weeks}"
        elif unit == "months":
            direct_ans = f"{diff_months}"
        else:
            direct_ans = f"{diff_days} days ago"

        cert_text = (
            f"[TEMPORAL CALCULATION CERTIFICATE]\n"
            f"- Event Evidence: {best_content[:180]}\n"
            f"- Event Date: {best_date.isoformat()}\n"
            f"- Question Reference Date: {ref_date.isoformat()}\n"
            f"- Verified Difference: {diff_days} days ({diff_weeks} weeks, ~{diff_months} months)\n"
            f"- Calculated Answer ({unit}): {direct_ans}\n"
            f"[/TEMPORAL CALCULATION CERTIFICATE]"
        )

        return TemporalCertificate(
            has_certificate=True,
            query_type="recency",
            event_date_str=best_date.isoformat(),
            reference_date_str=ref_str,
            difference_days=diff_days,
            difference_weeks=diff_weeks,
            difference_months=diff_months,
            ordered_events=None,
            formatted_certificate=cert_text,
            direct_answer_candidate=direct_ans,
        )

    def _resolve_ordering(self, question: str, records: Sequence[StructuredIR]) -> TemporalCertificate:
        """Extract candidate entities from question and order them by assertion date."""
        dates_and_events = []
        for r in records:
            d = parse_iso_date(r.time_scope or r.raw_content)
            if d:
                dates_and_events.append((d, r.raw_content))

        dates_and_events.sort(key=lambda x: x[0])
        if len(dates_and_events) < 2:
            return TemporalCertificate(
                has_certificate=False,
                query_type="ordering",
                event_date_str=None,
                reference_date_str=None,
                difference_days=None,
                difference_weeks=None,
                difference_months=None,
                ordered_events=None,
                formatted_certificate="",
                direct_answer_candidate=None,
            )

        earliest_date, earliest_content = dates_and_events[0]
        cert_text = (
            f"[CHRONOLOGICAL ORDERING CERTIFICATE]\n"
            f"- Earliest Date: {earliest_date.isoformat()}\n"
            f"- Earliest Event Evidence: {earliest_content[:200]}\n"
            f"[/CHRONOLOGICAL ORDERING CERTIFICATE]"
        )

        return TemporalCertificate(
            has_certificate=True,
            query_type="ordering",
            event_date_str=earliest_date.isoformat(),
            reference_date_str=None,
            difference_days=None,
            difference_weeks=None,
            difference_months=None,
            ordered_events=[(d.isoformat(), txt[:100]) for d, txt in dates_and_events[:5]],
            formatted_certificate=cert_text,
            direct_answer_candidate=None,
        )

    def _resolve_interval(self, question: str, records: Sequence[StructuredIR], unit: str) -> TemporalCertificate:
        """Resolve interval between two distinct events."""
        distinct_dates = sorted(list({
            parse_iso_date(r.time_scope or r.raw_content)
            for r in records
            if parse_iso_date(r.time_scope or r.raw_content)
        }))

        if len(distinct_dates) >= 2:
            d1, d2 = distinct_dates[0], distinct_dates[-1]
            diff_days = abs((d2 - d1).days)
            diff_weeks = diff_days // 7
            ans = f"{diff_days} days" if unit == "days" else f"{diff_weeks} weeks"

            cert_text = (
                f"[TEMPORAL INTERVAL CERTIFICATE]\n"
                f"- First Event Date: {d1.isoformat()}\n"
                f"- Second Event Date: {d2.isoformat()}\n"
                f"- Difference: {diff_days} days ({diff_weeks} weeks)\n"
                f"- Calculated Answer: {ans}\n"
                f"[/TEMPORAL INTERVAL CERTIFICATE]"
            )
            return TemporalCertificate(
                has_certificate=True,
                query_type="interval",
                event_date_str=d1.isoformat(),
                reference_date_str=d2.isoformat(),
                difference_days=diff_days,
                difference_weeks=diff_weeks,
                difference_months=None,
                ordered_events=None,
                formatted_certificate=cert_text,
                direct_answer_candidate=ans,
            )

        return TemporalCertificate(
            has_certificate=False,
            query_type="interval",
            event_date_str=None,
            reference_date_str=None,
            difference_days=None,
            difference_weeks=None,
            difference_months=None,
            ordered_events=None,
            formatted_certificate="",
            direct_answer_candidate=None,
        )

    def _find_best_date_for_phrase(self, phrase: str, records: Sequence[StructuredIR]) -> tuple[datetime.datetime | None, str]:
        """Find the earliest/best matching record date for a phrase or entity."""
        stop_words = {"the", "a", "an", "i", "my", "we", "our", "day", "for", "her", "his", "their", "stuff", "pick", "out", "of"}
        words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", phrase.lower())) - stop_words
        if not words:
            words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", phrase.lower()))

        best_date = None
        best_score = 0.0
        best_text = ""

        for r in records:
            d = self._extract_true_event_date(r.time_scope or r.raw_content, r.raw_content)
            if not d:
                continue
            r_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", r.raw_content.lower()))
            overlap = len(words & r_words)
            if overlap == 0:
                continue
            score = float(overlap)
            if (self._RE_REL_AGO.search(r.raw_content) or self._RE_MONTH_DAY.search(r.raw_content)
                    or "last" in r.raw_content.lower() or "yesterday" in r.raw_content.lower()):
                score += 2.5
            if score > best_score:
                best_score = score
                best_date = d
                best_text = r.raw_content

        return best_date, best_text

    def _resolve_pairwise_ordering(
        self,
        question: str,
        opt_a: str,
        opt_b: str,
        direction: str,
        records: Sequence[StructuredIR],
    ) -> TemporalCertificate:
        """Resolve pairwise ordering question between Option A and Option B."""
        d_a, txt_a = self._find_best_date_for_phrase(opt_a, records)
        d_b, txt_b = self._find_best_date_for_phrase(opt_b, records)

        if not d_a or not d_b:
            return self._resolve_ordering(question, records)

        is_asking_earlier = any(w in direction.lower() for w in ["first", "earlier"])
        if is_asking_earlier:
            winner = opt_a if d_a <= d_b else opt_b
            winner_date = min(d_a, d_b)
        else:
            winner = opt_a if d_a >= d_b else opt_b
            winner_date = max(d_a, d_b)

        # Allen Interval Algebra evaluation
        from artificial_memory.temporal.interval_algebra import TimeInterval
        ti_a = TimeInterval.point(d_a, opt_a)
        ti_b = TimeInterval.point(d_b, opt_b)
        rel = ti_a.relation_to(ti_b)
        allen_deduction = ti_a.format_derivation(ti_b)

        clean_winner = re.sub(r"^(?:the\s+)", "", winner.strip(), flags=re.I).capitalize()

        cert_text = (
            f"[CHRONOLOGICAL COMPARISON CERTIFICATE]\n"
            f"- Option A: '{opt_a}' (Date: {d_a.isoformat()})\n"
            f"- Option B: '{opt_b}' (Date: {d_b.isoformat()})\n"
            f"- Allen Interval Relation: {rel.value}\n"
            f"- Deduction: {allen_deduction}\n"
            f"- Target Ordering: {direction.upper()}\n"
            f"- Verified Result: {clean_winner} ({winner_date.isoformat()})\n"
            f"[/CHRONOLOGICAL COMPARISON CERTIFICATE]"
        )

        return TemporalCertificate(
            has_certificate=True,
            query_type="ordering",
            event_date_str=winner_date.isoformat(),
            reference_date_str=None,
            difference_days=abs((d_b - d_a).days),
            difference_weeks=abs((d_b - d_a).days) // 7,
            difference_months=None,
            ordered_events=[(d_a.isoformat(), opt_a), (d_b.isoformat(), opt_b)],
            formatted_certificate=cert_text,
            direct_answer_candidate=clean_winner,
            interval_relation=rel.value,
            allen_derivation=allen_deduction,
        )

    def _resolve_three_event_ordering(
        self,
        ev1: str,
        ev2: str,
        ev3: str,
        records: Sequence[StructuredIR],
    ) -> TemporalCertificate:
        """Resolve three-event chronological ordering."""
        d1, _ = self._find_best_date_for_phrase(ev1, records)
        d2, _ = self._find_best_date_for_phrase(ev2, records)
        d3, _ = self._find_best_date_for_phrase(ev3, records)

        events = [(d1 or datetime.date.min, ev1), (d2 or datetime.date.min, ev2), (d3 or datetime.date.min, ev3)]
        events.sort(key=lambda x: x[0])

        e1_name = events[0][1].strip()
        e2_name = events[1][1].strip()
        e3_name = events[2][1].strip()

        # Format: First, ..., then ..., and lastly, ...
        # Clean phrases: remove "the day " prefix if present to format natural speech
        def clean_ev(s: str) -> str:
            s_clean = s.strip()
            if s_clean.lower().startswith("the day "):
                s_clean = s_clean[8:]
            return s_clean

        from artificial_memory.temporal.interval_algebra import TimeInterval
        ti_1 = TimeInterval.point(events[0][0], e1_name)
        ti_2 = TimeInterval.point(events[1][0], e2_name)
        ti_3 = TimeInterval.point(events[2][0], e3_name)
        rel_12 = ti_1.relation_to(ti_2)
        rel_23 = ti_2.relation_to(ti_3)

        ans = f"First, {clean_ev(e1_name)}, then {clean_ev(e2_name)}, and lastly, {clean_ev(e3_name)}."
        cert_text = (
            f"[THREE EVENT ORDERING CERTIFICATE]\n"
            f"1. {events[0][0].isoformat()}: {e1_name}\n"
            f"2. {events[1][0].isoformat()}: {e2_name}\n"
            f"3. {events[2][0].isoformat()}: {e3_name}\n"
            f"- Allen Interval Sequence: {e1_name} ({rel_12.value}) -> {e2_name} ({rel_23.value}) -> {e3_name}\n"
            f"Ordered Sequence: {ans}\n"
            f"[/THREE EVENT ORDERING CERTIFICATE]"
        )

        return TemporalCertificate(
            has_certificate=True,
            query_type="ordering",
            event_date_str=events[0][0].isoformat(),
            reference_date_str=None,
            difference_days=None,
            difference_weeks=None,
            difference_months=None,
            ordered_events=[(d.isoformat(), name) for d, name in events],
            formatted_certificate=cert_text,
            direct_answer_candidate=ans,
            interval_relation=f"{rel_12.value},{rel_23.value}",
            allen_derivation=f"{ti_1.format_derivation(ti_2)} {ti_2.format_derivation(ti_3)}",
        )

    def _resolve_relative_anchor(
        self,
        question: str,
        records: Sequence[StructuredIR],
        ref_date: datetime.date,
        match: re.Match,
    ) -> TemporalCertificate:
        """Find historical turns anchored by a relative time expression."""
        target_date = None
        txt = match.group(0).lower()

        if "last " in txt:
            # Day of week
            days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            for idx, dname in enumerate(days_of_week):
                if dname in txt:
                    # target day of week
                    current_dow = ref_date.weekday()
                    diff_dow = (current_dow - idx) % 7
                    diff_dow = 7 if diff_dow == 0 else diff_dow
                    target_date = ref_date - datetime.timedelta(days=diff_dow)
                    break
        elif "weekend" in txt:
            # Most recent weekend
            current_dow = ref_date.weekday()
            # Sunday is 6, Saturday is 5
            diff = (current_dow - 6) % 7 if current_dow >= 0 else 1
            target_date = ref_date - datetime.timedelta(days=max(1, diff))
        else:
            num_str = match.group(1).lower() if match.group(1) else "1"
            num = self.WORD_TO_NUM.get(num_str)
            if num is None:
                try:
                    num = int(num_str)
                except ValueError:
                    num = 1
            unit = match.group(2).lower() if match.group(2) else "days"
            if "day" in unit:
                target_date = ref_date - datetime.timedelta(days=num)
            elif "week" in unit:
                target_date = ref_date - datetime.timedelta(weeks=num)
            elif "month" in unit:
                target_date = ref_date - datetime.timedelta(days=int(num * 30.4))
            elif "year" in unit:
                target_date = ref_date - datetime.timedelta(days=int(num * 365.25))

        if not target_date:
            return TemporalCertificate(
                has_certificate=False, query_type="date_anchor", event_date_str=None,
                reference_date_str=ref_date.isoformat(), difference_days=None,
                difference_weeks=None, difference_months=None, ordered_events=None,
                formatted_certificate="", direct_answer_candidate=None
            )

        # Find record closest to target_date
        best_record = None
        min_date_diff = 9999
        q_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", question.lower())) - set(match.group(0).lower().split())

        for r in records:
            d = parse_iso_date(r.time_scope or r.raw_content)
            if not d:
                continue
            date_diff = abs((d - target_date).days)
            # Allowed tolerance: 4 days for days/weeks, 10 days for months
            max_tol = 10 if "month" in txt else 5
            if date_diff <= max_tol:
                r_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", r.raw_content.lower()))
                overlap = len(q_words & r_words)
                if overlap >= 1 or date_diff < min_date_diff:
                    min_date_diff = date_diff
                    best_record = r

        if best_record:
            best_d = parse_iso_date(best_record.time_scope or best_record.raw_content)
            cert_text = (
                f"[DATE-ANCHORED EVIDENCE CERTIFICATE]\n"
                f"- Question Relative Anchor: '{match.group(0)}'\n"
                f"- Reference Date: {ref_date.isoformat()}\n"
                f"- Computed Target Date: {target_date.isoformat()}\n"
                f"- Matched Event Date: {best_d.isoformat() if best_d else 'Unknown'}\n"
                f"- Matched Event Details: {best_record.raw_content[:200]}\n"
                f"[/DATE-ANCHORED EVIDENCE CERTIFICATE]"
            )
            return TemporalCertificate(
                has_certificate=True,
                query_type="date_anchor",
                event_date_str=best_d.isoformat() if best_d else None,
                reference_date_str=ref_date.isoformat(),
                difference_days=min_date_diff,
                difference_weeks=min_date_diff // 7,
                difference_months=None,
                ordered_events=None,
                formatted_certificate=cert_text,
                direct_answer_candidate=None,
            )

        return TemporalCertificate(
            has_certificate=False, query_type="date_anchor", event_date_str=None,
            reference_date_str=ref_date.isoformat(), difference_days=None,
            difference_weeks=None, difference_months=None, ordered_events=None,
            formatted_certificate="", direct_answer_candidate=None
        )

    _RE_WHEN_QUERY = re.compile(
        r"\b(?:when|what (?:year|month|date|day)|which (?:year|month|date|day))\b",
        re.I
    )

    def _resolve_compound_query(
        self,
        question: str,
        records: Sequence[StructuredIR],
        ref_date: datetime.datetime | None,
        reference_date_str: str | None,
    ) -> TemporalCertificate:
        """Resolve absolute temporal query from relative dialogue mentions using Allen's AST algebra."""
        from artificial_memory.temporal.interval_algebra import (
            canonicalize_temporal_answer,
            parse_compound_temporal_expression,
            resolve_ast,
            _CANONICAL_MONTHS,
        )

        stop_words = {
            "when", "what", "which", "where", "who", "whom", "whose", "how", "did", "do", "does",
            "is", "are", "was", "were", "the", "a", "an", "i", "my", "you", "your", "he", "she",
            "his", "her", "they", "their", "year", "month", "date", "day", "in", "on", "at", "to",
            "first", "last", "ever", "have", "has", "had"
        }
        q_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", question.lower())) - stop_words
        if not q_words:
            return TemporalCertificate(
                has_certificate=False, query_type="compound", event_date_str=None,
                reference_date_str=reference_date_str, difference_days=None,
                difference_weeks=None, difference_months=None, ordered_events=None,
                formatted_certificate="", direct_answer_candidate=None
            )

        best_record = None
        best_score = 0.0
        best_sess_date = None

        for r in records:
            d = parse_iso_date(r.time_scope or r.raw_content)
            r_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", r.raw_content.lower()))
            overlap = len(q_words & r_words)
            if overlap == 0:
                continue
            score = float(overlap)
            if re.search(r"\b(?:ago|last|week before|month before|years?|months?|\d{4})\b", r.raw_content, re.I):
                score += 2.0
            if score > best_score:
                best_score = score
                best_record = r
                best_sess_date = d

        if not best_record or best_score < 2.0:
            return TemporalCertificate(
                has_certificate=False, query_type="compound", event_date_str=None,
                reference_date_str=reference_date_str, difference_days=None,
                difference_weeks=None, difference_months=None, ordered_events=None,
                formatted_certificate="", direct_answer_candidate=None
            )

        anchor_date = (best_sess_date or ref_date or datetime.datetime.now()).date()
        text = best_record.raw_content

        resolved_val = None
        matched_expr = ""

        # 1. Check for "week before <date>" or "week before" relative to session date
        m_week_before = re.search(r"\b(?:the\s+)?week before\s+([^\.,;!\n]+)", text, re.I)
        if m_week_before:
            anchor_candidate = m_week_before.group(1).strip()
            ast_node = parse_compound_temporal_expression(f"1 week before {anchor_candidate}", anchor_date)
            if ast_node:
                res_d = resolve_ast(ast_node)
                if isinstance(res_d, datetime.date):
                    resolved_val = f"{_CANONICAL_MONTHS[res_d.month].capitalize()} {res_d.year}"
                    matched_expr = m_week_before.group(0)
            elif "week before" in text.lower() and best_sess_date:
                t_target = best_sess_date.date() - datetime.timedelta(days=7)
                resolved_val = f"{_CANONICAL_MONTHS[t_target.month].capitalize()} {t_target.year}"
                matched_expr = "the week before"

        # 2. Check for "N years ago" (e.g. "3 years ago" -> anchor_year - N)
        if not resolved_val:
            m_yr_ago = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", text, re.I)
            if m_yr_ago:
                n_str = m_yr_ago.group(1).lower()
                n_val = self.WORD_TO_NUM.get(n_str, int(n_str) if n_str.isdigit() else 1)
                resolved_val = str(anchor_date.year - n_val)
                matched_expr = m_yr_ago.group(0)

        # 3. Check for "last year" -> anchor_year - 1
        if not resolved_val and re.search(r"\blast year\b", text, re.I):
            resolved_val = str(anchor_date.year - 1)
            matched_expr = "last year"

        # 4. Check for explicit 4-digit year in text (e.g. 2019, 2020)
        if not resolved_val:
            m_year = re.search(r"\b(19\d\d|20\d\d)\b", text)
            if m_year:
                resolved_val = m_year.group(1)
                matched_expr = f"in {resolved_val}"

        # 5. Check general parse_compound_temporal_expression across punctuation chunks
        if not resolved_val:
            for sent in re.split(r"[\.,;]", text):
                clean_s = sent.strip()
                if not clean_s:
                    continue
                ast_node = parse_compound_temporal_expression(clean_s, anchor_date)
                if ast_node:
                    res_eval = resolve_ast(ast_node)
                    if isinstance(res_eval, datetime.date):
                        resolved_val = canonicalize_temporal_answer(res_eval)
                        matched_expr = clean_s
                        break

        if resolved_val:
            clean_snippet = re.sub(r"\s+", " ", text)[:150].strip()
            cert_text = (
                f"[ALLEN TEMPORAL ALGEBRA CERTIFICATE]\n"
                f"- Event Mention: '{clean_snippet}'\n"
                f"- Anchor Date: {anchor_date.isoformat()}\n"
                f"- Temporal Expression: '{matched_expr}'\n"
                f"- Deterministic Calendar Deduction: {resolved_val}\n"
                f"[/ALLEN TEMPORAL ALGEBRA CERTIFICATE]"
            )
            return TemporalCertificate(
                has_certificate=True,
                query_type="compound",
                event_date_str=anchor_date.isoformat(),
                reference_date_str=reference_date_str,
                difference_days=None,
                difference_weeks=None,
                difference_months=None,
                ordered_events=None,
                formatted_certificate=cert_text,
                direct_answer_candidate=resolved_val,
                allen_derivation=f"Expression '{matched_expr}' relative to {anchor_date} deduces {resolved_val}",
            )

        return TemporalCertificate(
            has_certificate=False, query_type="compound", event_date_str=None,
            reference_date_str=reference_date_str, difference_days=None,
            difference_weeks=None, difference_months=None, ordered_events=None,
            formatted_certificate="", direct_answer_candidate=None
        )

