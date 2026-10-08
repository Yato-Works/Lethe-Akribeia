"""
Per-Session Strong Evidence Sampler for Multi-Session & Aggregation Queries.

Resolves the L2 selection bottleneck (where evidence is dropped across sessions
under budget constraints) by guaranteeing strong evidence representation
from each candidate session.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from artificial_memory.compiler.speaker_normalizer import SpeakerAttributionNormalizer
from artificial_memory.recall.evidence_scorer import UniversalEvidenceScorer


@dataclass(frozen=True)
class ScoredEvidenceTurn:
    session_id: str
    session_date: str
    speaker: str
    content: str
    score: float
    matched_terms: tuple[str, ...]


class PerSessionSampler:
    """Samples the strongest evidence turns per session for aggregation queries.

    v3 design:
    - turns_per_session=3: captures multi-fact sessions
    - min_score_threshold=1.0: quality bar; session-level gating (best turn must pass)
    - Typed-quantity bonus: CONDITIONAL on query term match (avoids noise from unrelated numbers)
    - User speaker boost: 1.5x (first-hand evidence > generic recommendations)
    - First-person quantity bonus: gold-standard evidence turns
    - Grouped session format: cleaner enumeration for the reader
    """

    STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "i", "you", "my", "your", "we", "our",
        "in", "on", "at", "to", "for", "with", "from", "by", "about", "of", "and", "or", "it",
        "this", "that", "there", "then", "did", "do", "does", "have", "had", "has", "me",
        "suggest", "recommend", "advice", "tips", "ideas", "some", "any", "please", "wondering",
        "tell", "show", "give", "help", "planning", "plans", "soon",
        "what", "which", "who", "whom", "whose", "how", "many", "much", "total", "combined",
    })

    _QUANTITY_RE = re.compile(
        r"(\$\d|\d+[\s-]*(dollars?|usd|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|miles?|km|"
        r"items?|pieces?|times?|visits?|trips?|books?|movies?|festivals?)|"
        r"\b(week-long|month-long|year-long|day-long)\b)",
        re.IGNORECASE,
    )

    _FIRST_PERSON_QTY_RE = re.compile(
        r"\b(i|we)\s+(spent|played|took|had|made|attended|visited|watched|read|"
        r"ran|hiked|camped|worked|exercised|traveled|bought|purchased|paid)\b",
        re.IGNORECASE,
    )

    _GENERIC_MATCHES = frozenset({
        "few", "past", "last", "recent", "recently", "month", "months",
        "amount", "total", "combined", "items", "item", "times", "time",
        "day", "days", "year", "years", "week", "weeks"
    })

    _KEY_SHORT_TOKENS = frozenset({"sf", "la", "ny", "tv", "ai", "vr", "pr"})

    def __init__(self, turns_per_session: int = 3, min_score_threshold: float = 1.0):
        self.turns_per_session = turns_per_session
        self.min_score_threshold = min_score_threshold
        self.speaker_normalizer = SpeakerAttributionNormalizer()

    @classmethod
    def extract_query_terms(cls, query: str) -> set[str]:
        """Extract meaningful lexical & entity search terms and expand via semantics."""
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower())
        terms = {
            w for w in words
            if (len(w) > 2 or any(c.isdigit() for c in w) or w in cls._KEY_SHORT_TOKENS)
            and w not in cls.STOP_WORDS
        }
        expanded = set(terms)
        for t in list(terms):
            if t in UniversalEvidenceScorer.SEMANTIC_EXPANSIONS:
                for exp in UniversalEvidenceScorer.SEMANTIC_EXPANSIONS[t]:
                    if len(exp) > 2 and exp not in cls.STOP_WORDS:
                        expanded.add(exp.lower())
        from artificial_memory.compiler.coverage.concept_expander import AbstractConceptExpander
        for c in AbstractConceptExpander.detect_concepts(query):
            tax = AbstractConceptExpander.CONCEPT_TAXONOMY[c]
            for group_terms in tax.values():
                for term in group_terms:
                    for w in term.split():
                        if len(w) > 2 and w.lower() not in cls.STOP_WORDS:
                            expanded.add(w.lower())

        # Morphological inflections (plurals, verb endings)
        inflected = set()
        for w in list(expanded):
            if w.endswith("ies") and len(w) > 4:
                inflected.add(w[:-3] + "y")
            elif w.endswith("es") and len(w) > 3:
                inflected.add(w[:-2])
            elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
                inflected.add(w[:-1])
            elif w.endswith("ing") and len(w) > 5:
                inflected.add(w[:-3])
            elif w.endswith("ed") and len(w) > 4:
                inflected.add(w[:-2])
                inflected.add(w[:-1])
        expanded.update(inflected)
        return expanded

    def score_turn(
        self,
        turn_content: str,
        query_terms: set[str],
        speaker: str = "user",
    ) -> tuple[float, list[str]]:
        """Score a turn for aggregation evidence relevance."""
        c_lower = turn_content.lower()
        turn_words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", c_lower))

        matched = []
        score = 0.0
        for t in query_terms:
            if " " in t:
                if re.search(r"\b" + re.escape(t) + r"\b", c_lower):
                    matched.append(t)
                    score += 3.5
            else:
                if t in turn_words:
                    matched.append(t)
                    score += 3.0 if len(t) > 4 else 1.5

        # Must match at least one non-generic topical term
        non_generic = [m for m in matched if m.lower() not in self._GENERIC_MATCHES]
        if not non_generic:
            return 0.0, []

        # Typed quantity bonus: CONDITIONAL on query term match (avoids noise)
        has_match = bool(matched)
        has_typed_qty = bool(self._QUANTITY_RE.search(turn_content))
        if has_match and has_typed_qty:
            score += 2.0

        # User speaker boost: first-hand evidence > generic recommendations
        if speaker.lower() == "user":
            score *= 2.0
            # Gold-standard: first-person action + typed quantity ("I spent 3 days camping")
            if has_match and has_typed_qty and self._FIRST_PERSON_QTY_RE.search(turn_content):
                score += 4.0

        return score, matched

    def sample(
        self,
        query: str,
        all_turns: Sequence[tuple[str, str, str, str]],
        max_total_tokens: int = 1800,
    ) -> list[ScoredEvidenceTurn]:
        """
        Extract the strongest evidence turns grouped by session.

        Session-level gating: a session is included if its best-scoring turn
        passes min_score_threshold.
        User turns are prioritized within each session to prevent assistant
        verbosity from displacing ground truth.
        Round-robin budget allocation ensures cross-session diversity.
        """
        query_terms = self.extract_query_terms(query)
        if not query_terms:
            return []

        sessions: dict[str, list[tuple[str, str, str, str]]] = {}
        for turn in all_turns:
            sid = turn[0]
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(turn)

        session_candidates: dict[str, list[ScoredEvidenceTurn]] = {}

        for sid, turns in sessions.items():
            user_scored: list[ScoredEvidenceTurn] = []
            asst_scored: list[ScoredEvidenceTurn] = []
            for t_sid, t_date, t_speaker, t_content in turns:
                sc, matched = self.score_turn(t_content, query_terms, speaker=t_speaker)
                if sc < self.min_score_threshold:
                    continue
                ev_turn = ScoredEvidenceTurn(
                    session_id=t_sid,
                    session_date=t_date,
                    speaker=t_speaker,
                    content=t_content,
                    score=sc,
                    matched_terms=tuple(matched),
                )
                if t_speaker.lower() == "user":
                    user_scored.append(ev_turn)
                else:
                    asst_scored.append(ev_turn)

            user_scored.sort(key=lambda x: x.score, reverse=True)
            asst_scored.sort(key=lambda x: x.score, reverse=True)

            # Clamp verbose assistant turns (>100 words) so they never crowd out user ground-truth facts
            clamped_asst: list[ScoredEvidenceTurn] = []
            for at in asst_scored:
                words = at.content.split()
                if len(words) > 100:
                    clamped_text = " ".join(words[:100]) + " ..."
                    clamped_asst.append(ScoredEvidenceTurn(
                        session_id=at.session_id, session_date=at.session_date, speaker=at.speaker,
                        content=clamped_text, score=at.score, matched_terms=at.matched_terms
                    ))
                else:
                    clamped_asst.append(at)

            selected_session_turns: list[ScoredEvidenceTurn] = []
            # User turns strictly first (ground truth facts)
            selected_session_turns.extend(user_scored[: self.turns_per_session])
            slots_left = self.turns_per_session - len(selected_session_turns)
            if slots_left > 0:
                selected_session_turns.extend(clamped_asst[:slots_left])
            elif clamped_asst and clamped_asst[0].score >= 12.0:
                # Strong assistant evidence turn alongside user turns
                selected_session_turns.append(clamped_asst[0])

            if selected_session_turns:
                session_candidates[sid] = selected_session_turns

        if not session_candidates:
            return []

        # Sort sessions by their top-scoring turn
        sorted_sids = sorted(
            session_candidates.keys(),
            key=lambda s: session_candidates[s][0].score,
            reverse=True,
        )

        # Priority packing:
        # Top-3 dominant sessions get their top 2-3 turns prioritized,
        # ensuring multi-item facts within the primary answer sessions are not
        # displaced by distractor sessions.
        rr_ordered_turns: list[ScoredEvidenceTurn] = []
        for s_idx, sid in enumerate(sorted_sids):
            if s_idx < 3:
                # Top session: add top 2 turns immediately
                rr_ordered_turns.extend(session_candidates[sid][:2])
            else:
                rr_ordered_turns.append(session_candidates[sid][0])

        # Second round: remaining turns
        for s_idx, sid in enumerate(sorted_sids):
            start_rank = 2 if s_idx < 3 else 1
            if len(session_candidates[sid]) > start_rank:
                rr_ordered_turns.extend(session_candidates[sid][start_rank:])

        packed_evidence: list[ScoredEvidenceTurn] = []
        total_tokens_est = 0
        for ev in rr_ordered_turns:
            tok_est = len(ev.content) // 4 + 15
            if total_tokens_est + tok_est <= max_total_tokens:
                packed_evidence.append(ev)
                total_tokens_est += tok_est

        packed_evidence.sort(key=lambda x: (x.session_date or "", x.session_id))
        return packed_evidence

    def format_as_context(
        self,
        evidences: Sequence[ScoredEvidenceTurn],
        provenance: bool = False,
    ) -> str:
        """Format sampled evidence grouped by session for clearer enumeration.

        ``provenance=True`` (condition R3) stamps every turn line with its
        source session id and date; the default keeps R0 output identical.
        """
        if not evidences:
            return "[NO MATCHING MULTI-SESSION EVIDENCE FOUND]"

        sessions_grouped: dict[str, list[ScoredEvidenceTurn]] = OrderedDict()
        for ev in evidences:
            key = f"{ev.session_date or ''}|{ev.session_id}"
            if key not in sessions_grouped:
                sessions_grouped[key] = []
            sessions_grouped[key].append(ev)

        n_sessions = len(sessions_grouped)
        n_turns = len(evidences)
        lines = [
            f"[EXHAUSTIVE MULTI-SESSION EVIDENCE TIMELINE: {n_sessions} sessions, {n_turns} relevant turns]"
        ]

        session_num = 0
        for key, evs in sessions_grouped.items():
            session_num += 1
            first = evs[0]
            date_str = f" ({first.session_date})" if first.session_date else ""
            # R3 provenance: the event header already carries the timestamp,
            # so only the source session id is missing -- stamp it once per
            # event (all turns in an event belong to the same session).
            sid_str = f" [{first.session_id}]" if provenance else ""
            lines.append(f"\n--- Event {session_num}{date_str}{sid_str} ---")
            for ev in evs:
                norm_content = self.speaker_normalizer.normalize_turn(ev.content, speaker=ev.speaker)
                lines.append(f"  {ev.speaker.capitalize()}: {norm_content}")

        return "\n".join(lines)

