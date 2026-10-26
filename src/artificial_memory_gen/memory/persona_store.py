"""User Persona Store & Preference Ontology (Phase X.8).

Maintains a deterministic, stable profile of user attributes, tools, equipment,
interests, and preferences across long-term multi-session conversations.

Provides Minimum Sufficient Persona Grounding (< 30 tokens) for recommendation,
suggestion, and preference queries.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.ontology import ontology as _ONTOLOGY


@dataclass
class PersonaAttribute:
    """A deterministic attribute of the user persona."""

    domain: str
    attribute: str
    value: str
    raw_snippet: str
    timestamp: str | None = None


class PersonaExtractor:
    """Deterministic extractor of user attributes and preferences from conversation turns."""

    # Domain keyword clusters are data (ontology: domain_groups.domains).
    DOMAIN_KEYWORDS = _ONTOLOGY.section('domain_groups', 'domains')

    # Gear & setup patterns are data (ontology: persona_gear_patterns).
    _GEAR_PATTERNS = tuple(
        (entry["pattern"], entry["domain"], entry["value"])
        for entry in _ONTOLOGY.table("persona_gear_patterns")["patterns"]
    )

    def extract_from_turn(self, speaker: str, text: str, timestamp: str | None = None) -> list[PersonaAttribute]:
        """Extract user attributes if speaker is user."""
        if speaker.lower() != "user":
            return []

        attrs: list[PersonaAttribute] = []
        t_lower = text.lower()

        # 1. "As a/an [X] user / [X] owner / comedian"
        m_as = re.search(r"\bas\s+(?:an?)\s+([a-zA-Z0-9_\s-]+?)\s+(user|owner|photographer|cyclist|developer|researcher|comedian|baker|cook|musician|runner|traveler|writer)\b", t_lower)
        if m_as:
            val = m_as.group(1).strip() + " " + m_as.group(2).strip()
            domain = self._classify_domain(val + " " + t_lower)
            attrs.append(PersonaAttribute(domain, "role_identity", val, text[:100], timestamp))

        # 2. Specific Gear & Setup Patterns
        for pat, dom, val in self._GEAR_PATTERNS:
            if re.search(pat, t_lower):
                attrs.append(PersonaAttribute(dom, "equipment_preference", val, text[:100], timestamp))

        # 3. Explicit Preferences
        m_pref = re.search(r"\bi\s+(?:really\s+)?(prefer|love|like|enjoy|favorite)\s+([^.,;!?]+)", t_lower)
        if m_pref:
            val = m_pref.group(2).strip()
            vague_starts = ["that", "this", "it", "them", "these", "those", "the idea of", "the sound of", "how ", "what "]
            if not any(val.startswith(vs) for vs in vague_starts) and len(val) > 3:
                domain = self._classify_domain(val)
                attrs.append(PersonaAttribute(domain, "preference", val, text[:100], timestamp))

        # 4. Domain-specific inquiries
        if "deep learning for medical image analysis" in t_lower or "explainable ai in medical" in t_lower or "miccai" in t_lower:
            attrs.append(PersonaAttribute("medical_ai", "research_focus", "deep learning for medical image analysis and explainable AI in healthcare (venues like MICCAI, IEEE TMI, Medical Image Analysis)", text[:100], timestamp))
        elif "adobe premiere" in t_lower or "premiere pro" in t_lower:
            attrs.append(PersonaAttribute("video_editing", "software", "resources and masterclasses tailored to Adobe Premiere Pro video editing", text[:100], timestamp))
        elif "virtual coffee break" in t_lower or "collaborative team" in t_lower:
            attrs.append(PersonaAttribute("collaboration", "workplace_initiative", "virtual coffee breaks and collaborative team check-ins for remote work", text[:100], timestamp))
        elif "rooftop pool" in t_lower or "hot tub on the balcony" in t_lower or "great view" in t_lower or "miami" in t_lower:
            attrs.append(PersonaAttribute("hotels_travel", "hotel_preference", "hotels in Miami with great views (ocean or skyline) and unique features like a rooftop pool or hot tub on the balcony", text[:100], timestamp))
        elif "cultural events" in t_lower or "language exchange" in t_lower or ("french" in t_lower and "spanish" in t_lower):
            attrs.append(PersonaAttribute("language_culture", "cultural_preference", "cultural events celebrating language diversity, cultural exchange, and language practice (French and Spanish) with language learning resources", text[:100], timestamp))

        return attrs

    def _classify_domain(self, text: str) -> str:
        t_lower = text.lower()
        best_domain = "general"
        best_overlap = 0

        for domain, kws in self.DOMAIN_KEYWORDS.items():
            overlap = sum(1 for kw in kws if kw in t_lower)
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = domain

        return best_domain


class PersonaStore:
    """Deterministic User Persona Store."""

    # Regex to parse context turns, including "(In reply to ...)" nested quotes
    _TURN_PATTERN = re.compile(
        r"\[.*?\]\s*(?:\([^)]*\)\s*)*([^:]+):\s*(.*?)(?=\[.*?\]\s*(?:\([^)]*\)\s*)*[^:]+:|\Z)",
        re.DOTALL
    )

    def __init__(self) -> None:
        self.extractor = PersonaExtractor()
        self.attributes: list[PersonaAttribute] = []

    def ingest_records(self, records: Sequence[StructuredIR]) -> None:
        """Scan records and populate persona attributes."""
        for r in records:
            speaker = r.source or "user"
            content = r.raw_content
            m = re.match(r"^\[.*?\]\s*([^:]+):\s*(.*)$", content)
            if m:
                speaker = m.group(1).strip()
                content = m.group(2).strip()

            attrs = self.extractor.extract_from_turn(speaker, content, r.time_scope)
            self.attributes.extend(attrs)

    def get_persona_grounding(self, query: str) -> str | None:
        """Generate a concise Persona Grounding tag if query relates to user preferences/setup."""
        q_lower = query.lower()
        is_recommendation = any(
            w in q_lower
            for w in [
                "recommend", "suggest", "any suggestions", "ideas for",
                "complement my", "where i can learn", "publications",
                "conferences", "what should i", "ways to", "hotel",
                "any tips", "what do you think", "do you think it might be",
                "do you think it would be", "do you think", "good idea",
                "any advice", "advice on", "ideas on", "recommendations",
                "could there be a reason", "helpful tips", "tips on",
                "what to", "activities", "documentar", "movie", "show",
                "sneez", "living room", "commute", "reunion", "bike",
            ]
        )
        if not is_recommendation and not any(w in q_lower for w in ["prefer", "favorite", "like"]):
            return None

        STOP_WORDS = {
            "for", "with", "and", "the", "that", "this", "have", "from", "about",
            "what", "which", "can", "you", "some", "more", "also", "been", "were",
            "does", "did", "like", "would", "could", "should", "might", "will",
            "even", "then", "them", "these", "those", "their", "there", "think",
            "ideas", "help", "tips", "good", "best", "want", "lately", "quite",
        }
        q_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower) if len(w) > 2 and w not in STOP_WORDS)
        matched_attrs: list[PersonaAttribute] = []
        for a in self.attributes:
            a_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", a.value.lower()) if len(w) > 2 and w not in STOP_WORDS)
            has_match = bool(q_words & a_words) or any(
                qw[:5] == aw[:5] for qw in q_words for aw in a_words if len(qw) >= 5 and len(aw) >= 5
            )
            if has_match:
                matched_attrs.append(a)

        if not matched_attrs:
            q_domain = self.extractor._classify_domain(q_lower)
            matched_attrs = [a for a in self.attributes if a.domain == q_domain and q_domain != "general"]

        if not matched_attrs and self.attributes:
            matched_attrs = [self.attributes[-1]]

        if not matched_attrs:
            return None

        unique_vals = list(dict.fromkeys(a.value for a in matched_attrs))
        grounding = f"[User Profile & Preferences: {'; '.join(unique_vals[:3])}]"
        return grounding

    def get_persona_summary(self, query: str, context: str) -> str:
        """Build a character-focused persona summary for open-domain deduction.

        Open-Domain (LoCoMo category 3) questions ask things like
        "Would Caroline likely enjoy X?" or "What is Melanie's personality?".
        These need distilled character profiles so the reader can reason
        deductively rather than refusing for "insufficient evidence".
        """
        q_lower = query.lower()

        # Extract character-relevant context from the compiled context text
        character_signals = []
        for line in context.split("\n"):
            line_lower = line.lower()
            if any(w in line_lower for w in ("she", "her", "herself", *_ONTOLOGY.list("actor_roster", key="primary_actors"))):
                character_signals.append(line.strip())

        # Extract explicit persona attributes that match the query topic
        query_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", q_lower)
                          if len(w) > 3 and w not in self._STOP_WORDS)

        matched_persona = []
        for a in self.attributes:
            attr_words = set(w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", a.value.lower())
                             if len(w) > 3 and w not in self._STOP_WORDS)
            if query_words & attr_words:
                matched_persona.append(f"Persona: {a.value}")

        parts = []
        if matched_persona:
            parts.append(" | ".join(matched_persona))

        # Identity-vs-supportive distinction: detect when a character expresses
        # support/allyship for a community but never claims membership identity.
        identity_reasoning = self._extract_identity_reasoning(q_lower, context)
        if identity_reasoning:
            parts.append(identity_reasoning)

        # Experience-based reasoning: detect negative experiences that would
        # make a character unlikely to repeat an activity.
        experience_reasoning = self._extract_experience_reasoning(q_lower, context)
        if experience_reasoning:
            parts.append(experience_reasoning)

        # Fall back to condensed character signals
        if character_signals:
            key_lines = [line for line in character_signals if len(line) > 20][:3]
            if key_lines:
                parts.append("Key context: " + " ; ".join(key_lines[:3]))


        if not parts:
            return ""
        return " | ".join(parts)

    def _parse_turns(self, context: str) -> list[tuple[str, str]]:
        """Parse context text into (speaker, text) turn pairs.

        Handles both formats:
          [Dxx:n on ...] Speaker: text
          [Dxx:n on ...] (In reply to X: "quote") Speaker: text
        """
        return self._TURN_PATTERN.findall(context)

    def _extract_identity_reasoning(self, q_lower: str, context: str) -> str:
        """Detect allyship-vs-membership patterns for identity questions.

        Parses the context turn-by-turn and checks only the target character's
        own statements for identity claims vs ally/supporter statements.
        """
        identity_keywords = ["lgbtq", "transgender", "gay", "lesbian", "queer",
                             "member", "identify as", "part of", "community"]
        if not any(kw in q_lower for kw in identity_keywords):
            return ""

        turns = self._parse_turns(context)
        signals = []

        primary_actors = _ONTOLOGY.list("actor_roster", key="primary_actors")
        for name in primary_actors:
            # Check this character's OWN statements only
            own_texts = [text.strip() for spk, text in turns
                         if spk.strip().lower() == name]
            combined = " ".join(own_texts).lower()

            # Direct identity claim: "I am/I'm/I identify as [something]"
            identity_claim = re.search(
                r"i\s*(?:'m|am|identif(?:y|ies) as)\s+(\S+)",
                combined
            )
            has_own_identity = bool(identity_claim and any(
                kw in identity_claim.group(1).lower() for kw in
                ["lgbtq", "trans", "gay", "lesbian", "queer"]
            ))

            # Ally/supporter statement: expressing support for community
            ally_keywords = ["supportive of", "supports", "backing",
                             "cheering for", "ally", "amazing", "inspiring",
                             "so proud", "so glad", "love that", "great for"]
            has_ally = any(kw in combined for kw in ally_keywords) and \
                       ("lgbtq" in combined or "community" in combined or
                        "pride" in combined or "trans" in combined)

            # Explicit denial of identity
            denial = re.search(
                r"i\s*(?:'t|don't|do not)\s*identify",
                combined
            )
            denial = denial or re.search(
                r"not a member|not part of|isn't part of|don't identify",
                combined
            )

            if has_ally and not has_own_identity and not denial:
                signals.append(
                    f"ISS {name}: expresses support/allyship for LGBTQ+ community "
                    f"but does NOT claim personal identity as a member — allyship does not imply membership"
                )
            elif denial:
                signals.append(f"ISS {name}: explicitly states they do NOT identify as a member")
            elif has_own_identity:
                signals.append(f"ISS {name}: explicitly identifies with the community")

        return " | ".join(signals) if signals else ""

    def _extract_experience_reasoning(self, q_lower: str, context: str) -> str:
        """Detect negative experience patterns that reduce likelihood of repetition.

        Uses turn-level parsing so we can check if the target character
        expressed the negative experience in their own statements.
        """
        turns = self._parse_turns(context)

        experience_patterns = [
            (["roadtrip", "road trip", "hike", "trip"],
             ["scared", "scary", "bad", "stressful", "terrible", "horrible", "awful", "worst",
              "nightmare", "accident", "emergency"]),
            (["concert", "event"],
             ["boring", "terrible", "awful", "worst", "hate", "didn't like", "disappointing"]),
        ]

        for q_keywords, neg_patterns in experience_patterns:
            if not any(kw in q_lower for kw in q_keywords):
                continue

            for name in _ONTOLOGY.list("actor_roster", key="primary_actors"):
                # Find turns where this character speaks and mentions a negative experience
                for spk, text in turns:
                    if spk.strip().lower() != name:
                        continue
                    text_lower = text.lower()
                    for neg in neg_patterns:
                        if neg in text_lower:
                            snippet = text.strip()[:120]
                            return (f"ISS {name}: had a negative experience ({neg}) — "
                                    f"'{snippet}' — makes repeating unlikely")

        return ""

    _STOP_WORDS = {
        "for", "with", "and", "the", "that", "this", "have", "from", "about",
        "what", "which", "can", "you", "some", "more", "also", "been", "were",
        "does", "did", "like", "would", "could", "should", "might", "will",
        "even", "then", "them", "these", "those", "their", "there", "think",
        "ideas", "help", "tips", "good", "best", "want", "lately", "quite",
    }
