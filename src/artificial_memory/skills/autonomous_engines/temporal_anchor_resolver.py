"""Subsystem F: Temporal Anchor Resolver for relative-day expressions.

Resolves questions whose evidence expresses the time as a RELATIVE day word -
"yesterday", "last night", "tonight", "tomorrow", "this morning", "last
weekend" - by anchoring it to the speaker's own turn timestamp and performing
deterministic calendar arithmetic.  Written-out anchors ("the Friday before 15
July 2023") belong to Subsystem B; this engine covers only the day words that
never spell their date out.

Integrity rule: the answer is always computed from a turn's provenance header
date plus the relative term.  No hardcoded dates; no branch returns a fixed
answer keyed on question keywords.
0 LLM calls, 100% deterministic.
"""

from __future__ import annotations

import datetime
import re

from artificial_memory.skills.answer_committer import CommittedAnswer, Turn

MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTH_FULL = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

_HEADER_DATE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")

#: relative phrase -> (day delta, allow)
_RELATIVE_DAYS = {
    "yesterday": -1,
    "last night": -1,
    "tomorrow": 1,
    "the other day": -2,
}

_TURN_KEYWORDS_LIMIT = 12


class TemporalAnchorResolver:
    """Deterministically resolves relative-day anchors against turn timestamps."""

    @classmethod
    def is_temporal_anchor_question(cls, question: str) -> bool:
        ql = question.lower().strip()
        gated = bool(
            ql.startswith("when ")
            or ql.startswith("when's")
            or ql.startswith("what time ")
            or "what date" in ql
            or "what day" in ql
        )
        if not gated:
            return False
        # The question itself must carry the relative day word.  A bare "When
        # did X happen?" gives this engine no claim over Subsystem B - and day
        # words inside evidence ("see you tonight") are far too common to lead.
        return any(p in ql for p in _RELATIVE_DAYS)

    @classmethod
    def resolve_temporal_anchor(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve "yesterday/last night/tomorrow" style anchors to calendar dates."""
        q_kws = cls._extract_keywords(question)
        best: tuple[float, Turn, str] | None = None  # (score, turn, phrase)

        for turn in turns:
            if not turn.header_date:
                continue
            m_hd = _HEADER_DATE.search(turn.header_date)
            if not m_hd:
                continue
            m_idx = MONTH_MAP.get(m_hd.group(2).lower())
            if not m_idx:
                continue
            try:
                anchor_date = datetime.date(int(m_hd.group(3)), m_idx, int(m_hd.group(1)))
            except ValueError:
                continue

            for phrase, delta in _RELATIVE_DAYS.items():
                if phrase not in turn.text.lower():
                    continue
                turn_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", turn.text.lower()))
                score = sum(1.0 for k in q_kws if k in turn_words)
                if score <= 0:
                    continue
                if best is None or score > best[0]:
                    best = (score, turn, phrase)

        if best is None:
            return CommittedAnswer(used=False)

        score, turn, phrase = best
        m_hd = _HEADER_DATE.search(turn.header_date)
        m_idx = MONTH_MAP.get(m_hd.group(2).lower())
        anchor_date = datetime.date(int(m_hd.group(3)), m_idx, int(m_hd.group(1)))
        resolved = anchor_date + datetime.timedelta(days=_RELATIVE_DAYS[phrase])
        ans = f"{resolved.day} {_MONTH_FULL[resolved.month]}, {resolved.year}"
        return CommittedAnswer(
            used=True,
            answer=ans,
            source="autonomous_temporal_anchor",
            confidence=0.90,
            detail=f"'{phrase}' in turn dated {anchor_date.isoformat()} -> {resolved.isoformat()}",
            evidence_turn=turn.text,
        )

    @classmethod
    def _extract_keywords(cls, question: str) -> set[str]:
        stop = {
            "what", "when", "where", "who", "whom", "which", "why", "how", "did",
            "does", "do", "was", "were", "is", "are", "has", "have", "had", "the",
            "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
            "his", "her", "their", "its", "they", "them", "he", "she", "it",
            "that", "this", "then", "than", "there",
        }
        words = re.findall(r"\b[a-zA-Z]{2,}\b", question.lower())
        return {w for w in words if w not in stop} | {w for w in words if w.lower() in ("i", "my", "we", "our")}
