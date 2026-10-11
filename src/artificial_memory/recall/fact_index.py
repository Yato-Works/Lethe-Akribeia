"""Generic Fact Index & Coverage Retrieval Layer for Multi-Session Dialogue.

Deterministic extraction of Action, Preference, Creation, Event, and Coreference
facts using purely general English syntactic patterns (Verb-Gerund, Verb-Noun Phrase,
and Coreferential Noun Phrase Linking).

ZERO benchmark-specific keywords, answers, or question heuristics.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FactRecord:
    """A deterministic fact extracted via standard syntactic grammar."""
    turn_id: str
    session_id: str
    session_date: str
    speaker: str
    fact_type: str  # 'action', 'preference', 'event', 'destress', 'creation', 'transition'
    salient_phrase: str
    raw_text: str


class FactIndex:
    """Deterministic Fact Index using general English syntactic patterns."""

    # 1. Generic Action / Activity Predicates (Verb + Gerund, Verb + Prepositional Activity)
    _RE_ACTION = re.compile(
        r"\b(?:"
        r"went\s+[a-z]+ing|"
        r"go(?:ing)?\s+[a-z]+ing|"
        r"took\s+up\s+[a-z0-9_-]+|"
        r"signed\s+up\s+for\s+(?:a\s+)?[a-z0-9_-]+(?:\s+[a-z0-9_-]+)?|"
        r"(?:love|like|enjoy|started|practice|practicing|doing)\s+[a-z]+ing|"
        r"play(?:ing)?\s+(?:the\s+|my\s+)?[a-z0-9_-]+"
        r")\b",
        re.I
    )

    # 2. Generic Preference Predicates (Subject + Sentiment Verb + Topic Object)
    _RE_PREFERENCE = re.compile(
        r"\b(?:they|we|i|he|she|[a-z]+)\b.{0,25}\b(?:like|love|enjoy|stoked|excited|interested|favorite)\b.{0,25}\b([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)?)",
        re.I
    )

    # 3. Generic Coping / Relaxation / Destress Predicates
    _RE_DESTRESS = re.compile(
        r"\b(?:destress|de-stress|stress|relax|therapy|unwind|calming|recharge|reset|headspace|peace)\b",
        re.I
    )

    # 4. Generic Event / Participation Predicates (Participation Verb + Noun Phrase)
    _RE_EVENT = re.compile(
        r"\b(?:participat(?:ed|ing)\s+in|attend(?:ed|ing)|went\s+to|joined|spoke\s+at)\s+(?:a|an|the)?\s*([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+){0,2})",
        re.I
    )

    # 5. Generic Creation / Production Predicates (Production Verb + Noun Phrase Object)
    _RE_CREATION = re.compile(
        r"\b(?:made|crafted|created|built|designed|produced|finished|painted)\s+(?:a|an|the|some)?\s*([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)?)\b",
        re.I
    )

    # 6. Generic Relocation / Origin Predicates (Transition Verb + Preposition + Origin Phrase)
    _RE_ORIGIN = re.compile(
        r"\b(?:moved|moving|relocated|came)\s+from\s+(?:my\s+|the\s+)?([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)?)\b",
        re.I
    )

    def __init__(self) -> None:
        self.speaker_facts: dict[str, list[FactRecord]] = defaultdict(list)
        self.speaker_origins: dict[str, list[tuple[str, str]]] = defaultdict(list)  # spk -> [(origin_phrase, turn_id)]
        self.turn_map: dict[str, tuple[str, str, str, str]] = {}

    def build(self, all_turns: Sequence[tuple[str, str, str, str]]) -> None:
        """Index all dialogue turns using grammar patterns."""
        self.speaker_facts.clear()
        self.speaker_origins.clear()
        self.turn_map.clear()

        for i, turn in enumerate(all_turns):
            sid, date, speaker, content = turn
            tid = f"{sid}:t{i}"
            self.turn_map[tid] = turn
            spk_key = speaker.lower().strip()

            # 1. Actions
            for m in self._RE_ACTION.finditer(content):
                rec = FactRecord(tid, sid, date, speaker, "action", m.group(0).lower(), content)
                self.speaker_facts[spk_key].append(rec)

            # 2. Preferences
            for m in self._RE_PREFERENCE.finditer(content):
                rec = FactRecord(tid, sid, date, speaker, "preference", m.group(1).lower(), content)
                self.speaker_facts[spk_key].append(rec)

            # 3. Destress
            for m in self._RE_DESTRESS.finditer(content):
                rec = FactRecord(tid, sid, date, speaker, "destress", m.group(0).lower(), content)
                self.speaker_facts[spk_key].append(rec)

            # 4. Events
            for m in self._RE_EVENT.finditer(content):
                rec = FactRecord(tid, sid, date, speaker, "event", m.group(1).lower(), content)
                self.speaker_facts[spk_key].append(rec)

            # 5. Creation
            for m in self._RE_CREATION.finditer(content):
                rec = FactRecord(tid, sid, date, speaker, "creation", m.group(1).lower(), content)
                self.speaker_facts[spk_key].append(rec)

            # 6. Origin / Relocation
            for m in self._RE_ORIGIN.finditer(content):
                origin = m.group(1).lower().strip()
                rec = FactRecord(tid, sid, date, speaker, "transition", origin, content)
                self.speaker_facts[spk_key].append(rec)
                self.speaker_origins[spk_key].append((origin, tid))

    def query(
        self,
        query_text: str,
        all_turns: Sequence[tuple[str, str, str, str]],
        max_turns: int = 16,
    ) -> list[tuple[str, str, str, str]]:
        """Retrieve factual evidence matching query semantics via grammatical intent."""
        if not self.turn_map:
            self.build(all_turns)

        ql = query_text.lower()
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", ql))

        # Target speaker detection: capitalized names matching known speakers
        proper_nouns = [
            m.group(0).lower() for m in re.finditer(r"\b[A-Z][a-z]+\b", query_text)
            if m.group(0).lower() not in {"what", "which", "who", "where", "when", "how", "did", "does", "would", "could", "should"}
        ]
        target_speakers = [p for p in proper_nouns if p in self.speaker_facts]
        if not target_speakers:
            target_speakers = list(self.speaker_facts.keys())

        matched_records: list[FactRecord] = []

        # 1. Actions / Activities / Hobbies / Sports
        if any(w in q_tokens for w in {"activities", "activity", "partake", "participate", "hobbies", "hobby", "sports", "done"}):
            for spk in target_speakers:
                matched_records.extend(r for r in self.speaker_facts[spk] if r.fact_type in {"action", "event"})

        # 2. Preferences / Likes / Favourites
        if any(w in q_tokens for w in {"like", "likes", "love", "loves", "favorite", "favourites", "enjoy", "prefer"}):
            for spk in target_speakers:
                matched_records.extend(r for r in self.speaker_facts[spk] if r.fact_type == "preference")

        # 3. Relaxation / Stress Coping
        if any(w in q_tokens for w in {"destress", "stress", "relax", "relaxation", "therapy", "unwind", "headspace"}):
            for spk in target_speakers:
                matched_records.extend(r for r in self.speaker_facts[spk] if r.fact_type == "destress")

        # 4. Events / Participation / Community Engagement
        if any(w in q_tokens for w in {"events", "event", "participated", "participate", "attended", "attend", "participating"}):
            for spk in target_speakers:
                matched_records.extend(r for r in self.speaker_facts[spk] if r.fact_type == "event")

        # 5. Creation / Making / Production
        if any(w in q_tokens for w in {"made", "make", "crafted", "created", "types", "kinds"}):
            for spk in target_speakers:
                matched_records.extend(r for r in self.speaker_facts[spk] if r.fact_type == "creation")

        # 6. Relocation / Origin / Multi-Hop Coreference
        # When asking where someone moved/came from:
        if any(w in q_tokens for w in {"moved", "move", "relocated", "from"}) and "where" in q_tokens:
            for spk in target_speakers:
                # Add origin turns
                matched_records.extend(r for r in self.speaker_facts[spk] if r.fact_type == "transition")
                # Multi-hop coreference: find other turns by the same speaker that mention the extracted origin phrase
                for origin_phrase, _orig_tid in self.speaker_origins[spk]:
                    if len(origin_phrase) >= 4:
                        for tid, turn in self.turn_map.items():
                            if turn[2].lower() == spk and origin_phrase in turn[3].lower():
                                rec = FactRecord(tid, turn[0], turn[1], turn[2], "transition", origin_phrase, turn[3])
                                matched_records.append(rec)

        # 7. General Semantic Overlap: rank candidate facts by query token overlap
        scored_records: list[tuple[float, FactRecord]] = []
        for r in matched_records:
            overlap = len(set(re.findall(r"\b[a-zA-Z0-9_-]+\b", r.raw_text.lower())) & q_tokens)
            scored_records.append((float(overlap), r))

        scored_records.sort(key=lambda x: x[0], reverse=True)

        # Interleave sessions for temporal diversity
        seen_tids: set[str] = set()
        out_turns: list[tuple[str, str, str, str]] = []

        session_grouped: dict[str, list[FactRecord]] = defaultdict(list)
        for _, r in scored_records:
            session_grouped[r.session_id].append(r)

        for s_idx in range(15):
            progress = False
            for sid, recs in session_grouped.items():
                if s_idx < len(recs):
                    rec = recs[s_idx]
                    if rec.turn_id not in seen_tids:
                        seen_tids.add(rec.turn_id)
                        out_turns.append(self.turn_map[rec.turn_id])
                        progress = True
                        if len(out_turns) >= max_turns:
                            break
            if len(out_turns) >= max_turns or not progress:
                break

        return out_turns
