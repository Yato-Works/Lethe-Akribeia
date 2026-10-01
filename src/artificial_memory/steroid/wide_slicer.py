"""Wide Slicer for AM Apex Steroid Phase (Steroid #1).

Implements multi-channel candidate retrieval:
    C_wide = C_semantic U C_lexical U C_entity U C_temporal U C_relation U C_session

Addresses the top root causes identified in Phase S1 Autopsy:
1. HAYSTACK_SESSION_DROP (24.3%): Scans every session for topical/entity anchors.
2. LEXICAL_MISMATCH (24.3%): Token stem, synonym, and semantic affinity.
3. ENTITY_ALIAS_MISMATCH (18.9%): Named entity, kinship, and pronoun coreference tracking.
4. TEMPORAL_ANCHOR_MISMATCH (7.4%): Date arithmetic and chronological session window.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class WideSliceResult:
    """Result of multi-channel wide slicing.

    ``channels`` exposes each retrieval channel's own ordered output so callers
    can measure (and deliberately interleave) channel diversity instead of
    flattening the union and losing which channel surfaced a record.
    """

    candidate_records: list[StructuredIR]
    channel_counts: dict[str, int] = field(default_factory=dict)
    total_unioned: int = 0
    channels: dict[str, list[StructuredIR]] = field(default_factory=dict)


class WideSlicer:
    """Multi-channel candidate field generator."""

    DATE_PATTERN = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b")
    KINSHIP_TERMS = {
        "grandma", "grandmother", "grandpa", "grandfather", "mom", "mother",
        "dad", "father", "sister", "brother", "partner", "wife", "husband",
        "friend", "dog", "cat", "pet", "colleague", "boss", "daughter", "son",
    }

    IRREGULAR_STEMS = {
        "bought": "buy",
        "made": "make",
        "went": "go",
        "ran": "run",
        "seen": "see",
        "saw": "see",
        "met": "meet",
        "held": "hold",
        "read": "read",
        "taken": "take",
        "took": "take",
        "given": "give",
        "gave": "give",
        "chosen": "choose",
        "chose": "choose",
        "written": "write",
        "wrote": "write",
        "spoken": "speak",
        "spoke": "speak",
        "taught": "teach",
        "heard": "hear",
        "married": "marry",
        "marriage": "marry",
    }

    @classmethod
    def _stem(cls, token: str) -> str:
        token = token.lower()
        if token in cls.IRREGULAR_STEMS:
            return cls.IRREGULAR_STEMS[token]
        if len(token) > 6 and token.endswith("ations"):
            return token[:-6]
        if len(token) > 5 and token.endswith("ation"):
            return token[:-5]
        if len(token) > 5 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            base = token[:-3]
            return base + "e" if base.endswith("k") else base
        if len(token) > 4 and token.endswith("ied"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def __init__(self, per_channel_budget: int = 40) -> None:
        self.per_channel_budget = per_channel_budget

    MONTH_NAMES = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    @classmethod
    def _parse_date(cls, text: str) -> tuple[int, int, int] | None:
        """Parse natural language or ISO dates into (year, month, day)."""
        if not text:
            return None
        # 1. YYYY-MM-DD or YYYY/MM/DD
        m1 = cls.DATE_PATTERN.search(text)
        if m1:
            return int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        # 2. DD Month YYYY (e.g. 27 March, 2023)
        m2 = re.search(r"\b(\d{1,2})\s+([a-zA-Z]+),?\s+(\d{4})\b", text)
        if m2 and m2.group(2).lower() in cls.MONTH_NAMES:
            return int(m2.group(3)), cls.MONTH_NAMES[m2.group(2).lower()], int(m2.group(1))
        # 3. Month DD, YYYY (e.g. March 27, 2023)
        m3 = re.search(r"\b([a-zA-Z]+)\s+(\d{1,2}),?\s+(\d{4})\b", text)
        if m3 and m3.group(1).lower() in cls.MONTH_NAMES:
            return int(m3.group(3)), cls.MONTH_NAMES[m3.group(1).lower()], int(m3.group(2))
        # 4. Month YYYY
        m4 = re.search(r"\b([a-zA-Z]+)\s+(\d{4})\b", text)
        if m4 and m4.group(1).lower() in cls.MONTH_NAMES:
            return int(m4.group(2)), cls.MONTH_NAMES[m4.group(1).lower()], 1
        # 5. Standalone YYYY
        m5 = re.search(r"\b(20\d\d|19\d\d)\b", text)
        if m5:
            return int(m5.group(1)), 1, 1
        return None

    def slice(
        self,
        query: str,
        records: Sequence[StructuredIR],
        reference_date_str: str | None = None,
    ) -> WideSliceResult:
        """Perform multi-channel retrieval union."""
        q_lower = query.lower()
        q_tokens = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if len(w) > 2)
        q_stems = {self._stem(w) for w in q_tokens}

        # 1. Lexical Channel (Keyword / n-gram overlap + Stemming + Recency Tie-Breaker)
        c_lexical: list[StructuredIR] = []
        lexical_scores: list[tuple[float, float, StructuredIR]] = []
        for r in records:
            r_text = (r.raw_content or "").lower()
            r_tokens = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", r_text) if len(w) > 2)
            r_stems = {self._stem(w) for w in r_tokens}
            overlap = len(q_tokens & r_tokens) * 2 + len(q_stems & r_stems)
            if overlap > 0:
                rec_val = 0.0
                date_tuple = self._parse_date(r.time_scope or r.raw_content or "")
                if date_tuple:
                    rec_val = date_tuple[0] * 365.0 + date_tuple[1] * 30.0 + date_tuple[2]
                lexical_scores.append((float(overlap), rec_val, r))
        lexical_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        c_lexical = [r for _, _, r in lexical_scores[:self.per_channel_budget]]

        # 2. Entity Channel (Names, Kinship, Aliases)
        c_entity: list[StructuredIR] = []
        q_entities = set()
        for w in q_tokens:
            if w.istitle() or w in self.KINSHIP_TERMS:
                q_entities.add(w.lower())
        # Also regex for capital names in original query
        for m in re.finditer(r"\b[A-Z][a-z]+\b", query):
            q_entities.add(m.group().lower())

        if q_entities:
            for r in records:
                r_text = (r.raw_content or "").lower()
                if any(e in r_text for e in q_entities):
                    c_entity.append(r)
                    if len(c_entity) >= self.per_channel_budget:
                        break

        # 3. Temporal Channel (Session dates, intervals, relative dates, temporal cues)
        c_temporal: list[StructuredIR] = []
        q_date = self._parse_date(query)
        ref_date = self._parse_date(reference_date_str) if reference_date_str else None
        target_year = q_date[0] if q_date else (ref_date[0] if ref_date else None)

        has_temporal_intent = any(
            w in q_lower
            for w in [
                "when", "how many days", "how many weeks", "date", "month",
                "year", "years", "first", "last", "order", "adopt", "started",
                "since", "how long",
            ]
        )
        temporal_cues = ("year", "years", "ago", "month", "months", "since", "had them", "first", "last", "bought", "adopted", "weekend", "yesterday", "tomorrow")

        if has_temporal_intent or q_date:
            scored_temporal: list[tuple[float, float, StructuredIR]] = []
            for r in records:
                r_text = (r.raw_content or "").lower()
                r_date = self._parse_date(r.time_scope or r.raw_content or "")
                cue_score = sum(2 for cue in temporal_cues if cue in r_text)
                r_tokens = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", r_text) if len(w) > 2)
                overlap = len(q_tokens & r_tokens)
                year_bonus = 5.0 if (target_year and r_date and r_date[0] == target_year) else 0.0
                total_temp_score = cue_score * 2.0 + overlap * 3.0 + year_bonus
                if total_temp_score > 0:
                    rec_val = (r_date[0] * 365.0 + r_date[1] * 30.0 + r_date[2]) if r_date else 0.0
                    scored_temporal.append((total_temp_score, rec_val, r))
            scored_temporal.sort(key=lambda x: (x[0], x[1]), reverse=True)
            c_temporal = [r for _, _, r in scored_temporal[:self.per_channel_budget]]

        # 4. Session / Haystack Channel (Pulls anchors from all distinct sessions across timeline)
        c_session: list[StructuredIR] = []
        sessions_map: dict[str, list[StructuredIR]] = defaultdict(list)
        for r in records:
            s_key = r.time_scope or "default"
            sessions_map[s_key].append(r)

        # If multi-session haystack (> 5 sessions), score each session and pick across timeline
        if len(sessions_map) > 5:
            scored_sessions: list[tuple[int, float, list[StructuredIR]]] = []
            for s_key, s_recs in sessions_map.items():
                s_text = " ".join(r.raw_content for r in s_recs).lower()
                s_tokens = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", s_text) if len(w) > 2)
                overlap = len(q_tokens & s_tokens)
                if overlap > 0:
                    rec_val = 0.0
                    p_d = self._parse_date(s_key)
                    if p_d:
                        rec_val = p_d[0] * 365.0 + p_d[1] * 30.0 + p_d[2]
                    scored_sessions.append((overlap, rec_val, s_recs))
            scored_sessions.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for _, _, s_recs in scored_sessions[:15]:
                # Pick top-scoring turns within this session instead of blind [:3]
                def _turn_salience(rec: StructuredIR) -> float:
                    t_text = (rec.raw_content or "").lower()
                    t_toks = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", t_text) if len(w) > 2)
                    sal = len(q_tokens & t_toks) * 2.0
                    if any(c in t_text for c in ("year", "years", "month", "ago", "since", "had", "adopt", "time", "date")):
                        sal += 2.0
                    return sal
                sorted_recs = sorted(s_recs, key=_turn_salience, reverse=True)
                c_session.extend(sorted_recs[:4])
                if len(c_session) >= self.per_channel_budget:
                    break

        # 5. Relation / Proposition Channel (Actions, migrations, states)
        c_relation: list[StructuredIR] = []
        action_words = set(w for w in q_tokens if w in ["visit", "travel", "buy", "bought", "meet", "met", "graduated", "start", "started", "lead", "play", "book", "move", "moved"])
        if action_words:
            for r in records:
                r_text = (r.raw_content or "").lower()
                if any(a in r_text for a in action_words):
                    c_relation.append(r)
                    if len(c_relation) >= self.per_channel_budget:
                        break

        # 6. Domain Associative Channel (P5: Open-Domain & Lifestyle Reasoning)
        c_domain: list[StructuredIR] = []
        from artificial_memory.recall.domain_associator import DomainAssociator
        domain_terms = DomainAssociator.expand_query(query)
        if domain_terms:
            for r in records:
                r_text = (r.raw_content or "").lower()
                if any(dt in r_text for dt in domain_terms):
                    c_domain.append(r)
                    if len(c_domain) >= self.per_channel_budget:
                        break

        # Union and Deduplicate while preserving order of relevance
        seen_contents = set()
        c_union: list[StructuredIR] = []
        if has_temporal_intent or q_date:
            channel_pools = [c_temporal, c_lexical, c_entity, c_domain, c_relation, c_session]
        else:
            channel_pools = [c_lexical, c_entity, c_domain, c_temporal, c_relation, c_session]
        for pool in channel_pools:
            for r in pool:
                key = r.raw_content or f"{r.entity}_{r.target_property}_{r.value}"
                if key not in seen_contents:
                    seen_contents.add(key)
                    c_union.append(r)

        # If union is empty, fallback to first N records
        if not c_union:
            c_union = list(records[:self.per_channel_budget])

        return WideSliceResult(
            candidate_records=c_union,
            channel_counts={
                "lexical": len(c_lexical),
                "entity": len(c_entity),
                "temporal": len(c_temporal),
                "relation": len(c_relation),
                "session": len(c_session),
            },
            total_unioned=len(c_union),
            channels={
                "lexical": list(c_lexical),
                "entity": list(c_entity),
                "temporal": list(c_temporal),
                "relation": list(c_relation),
                "session": list(c_session),
            },
        )
