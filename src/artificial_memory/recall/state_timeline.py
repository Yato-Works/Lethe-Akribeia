"""State Timeline Engine for Lethe-Akribeia.

Constructs explicit chronological state evolution timelines from ACTUAL records:

    Timeline(e, p) = [(t1, v1), (t2, v2), ...]

Determines whether the query targets the Current/Latest state or a Previous
state, and generates a clean, proof-carrying state timeline certificate.

CLEAN-R7 (2026-10): the previous ``SURGICAL_CERT_MAP`` — a hand-written table
of benchmark answers keyed by topic string (e.g. a footwear-storage topic
mapped to its gold answer) — was deleted as benchmark overfitting.  It
returned a fixed certificate WITHOUT ever reading the conversation records,
so it could never generalise beyond the development split.  This engine now
derives every statement from the retrieved records and their dates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class StateTimelineGrounding:
    """Grounding certificate produced by StateTimelineEngine."""
    certificate: str
    target_state: str  # "current" or "previous"


def _parse_date(text: str) -> datetime | None:
    """Parse the first ISO-like date in ``text``; None if absent."""
    if not text:
        return None
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", text)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


class StateTimelineEngine:
    """Resolves knowledge-update queries by building state timelines from records.

    Fully general: it never references any benchmark topic.  It groups records
    that mention the query's salient entity, orders them by date, and emits the
    earliest (previous) or latest (current) observed state.
    """

    # Generic temporal markers.  A question using one of these asks for the
    # PREVIOUS state rather than the latest.
    PREVIOUS_MARKERS = (
        "previous", "previously", "earlier", "before", "former",
        "initially", "used to", "at first", "originally",
    )

    def build_timeline_certificate(
        self,
        question: str,
        records: Sequence[StructuredIR],
    ) -> StateTimelineGrounding | None:
        """Build an explicit state timeline certificate from records."""
        ql = question.lower()
        is_previous = any(w in ql for w in self.PREVIOUS_MARKERS)

        # Salient terms from the question: content words that can anchor a
        # state.  Deliberately generic — no benchmark vocabulary.
        q_words = self._salient_terms(ql)
        if not q_words:
            return None

        # Collect dated observations that mention the salient terms.
        observations: list[tuple[datetime, str, str]] = []
        for r in records:
            text = r.raw_content or r.value or ""
            tl = text.lower()
            if not any(w in tl for w in q_words):
                continue
            dt = _parse_date(getattr(r, "time_scope", "") or "") or _parse_date(text)
            if dt is None:
                continue
            observations.append((dt, text.strip(), getattr(r, "time_scope", "") or dt.isoformat()))

        if len(observations) < 2:
            # A timeline needs at least two points to speak of an update.
            return None

        observations.sort(key=lambda x: x[0])
        first_dt, first_txt, first_scope = observations[0]
        last_dt, last_txt, last_scope = observations[-1]

        if is_previous:
            target_dt, target_txt, target_scope = first_dt, first_txt, first_scope
            target_state = "previous"
            cert = (
                "[STATE TIMELINE & EVOLUTION]\n"
                f"1. Earlier / Initial State ({first_scope}): {self._clip(first_txt)}\n"
                f"2. Later / Updated State ({last_scope}): {self._clip(last_txt)}\n\n"
                "[INSTRUCTION: The question asks for the EARLIER state. "
                "Answer strictly from the earlier state above.]"
            )
        else:
            target_dt, target_txt, target_scope = last_dt, last_txt, last_scope
            target_state = "current"
            cert = (
                "[STATE TIMELINE & EVOLUTION]\n"
                f"1. Earlier State ({first_scope}): {self._clip(first_txt)}\n"
                f"2. Updated / Latest State ({last_scope}): {self._clip(last_txt)}\n\n"
                "[INSTRUCTION: The question asks for the CURRENT state. "
                "Answer strictly from the latest state above.]"
            )

        return StateTimelineGrounding(certificate=cert, target_state=target_state)

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #

    _STOP = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
        "i", "my", "me", "you", "your", "what", "where", "when", "who",
        "how", "many", "much", "of", "to", "in", "on", "at", "for", "and",
        "or", "it", "that", "this", "with", "now", "currently", "keep",
        "get", "got", "have", "has", "had",
    })

    def _salient_terms(self, ql: str) -> list[str]:
        """Content words from the question, longest first, stop-words removed."""
        words = re.findall(r"[a-z0-9']+", ql)
        terms = [w for w in words if len(w) >= 4 and w not in self._STOP]
        # de-dup, preserve a deterministic order (longest first)
        seen: set[str] = set()
        out: list[str] = []
        for w in sorted(terms, key=lambda x: (-len(x), x)):
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    @staticmethod
    def _clip(text: str, limit: int = 240) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
