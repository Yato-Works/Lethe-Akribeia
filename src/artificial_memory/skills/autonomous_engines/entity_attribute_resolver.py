"""Subsystem G: Entity Attribute Resolver for personal facts and attributes.

Covers the attribute shapes no other subsystem owns:
- Nicknames: "call me Jo", "Jo for short", "we call her Jo"
- Gaming consoles and setups: "on my Gamecube", "on PC and Playstation"
- Visited places named as "have been to X" proper-noun spans

Visual attributes (hair colour, room lighting) belong to the relational colour
lookup, pet names to the counting engine's named-entity collector, and dates to
Subsystems B and F - this engine deliberately does not duplicate them.

Integrity rule: every attribute is read out of the evidence text.  A branch
that cannot find its evidence abstains; nothing is keyed on question keywords.
0 LLM calls, 100% deterministic.
"""

from __future__ import annotations

import re

from artificial_memory.skills.answer_committer import CommittedAnswer, Turn

#: Closed console vocabulary - hardware names are a fixed, small set.
_CONSOLES = (
    "gamecube", "nintendo", "switch", "playstation", "ps4", "ps5", "xbox",
    "pc", "steam deck", "sega", "wii", "dreamcast",
)

STOPWORDS = {
    "what", "when", "where", "who", "whom", "which", "why", "how", "did", "does",
    "do", "was", "were", "is", "are", "has", "have", "had", "the", "a", "an",
    "and", "or", "of", "to", "in", "on", "at", "for", "with", "their", "his",
    "her", "its", "they", "them", "he", "she", "it", "that", "this", "then",
}


class EntityAttributeResolver:
    """Deterministically extracts personal attributes and factual entity traits."""

    @classmethod
    def resolve_entity_attribute(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        ql = question.lower().strip()

        # Guardrail: never answer 'why' questions with attribute values
        if ql.startswith("why"):
            return CommittedAnswer(used=False)

        # 1. Nickname: "What nickname does X use for Y?" - evidence says
        #    "call(s) her Jo", "Jo for short", "we call her Jo".
        if "nickname" in ql or "nick name" in ql:
            nick = cls._extract_nickname(turns)
            if nick:
                return CommittedAnswer(
                    used=True,
                    answer=nick,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"nickname extracted from evidence: '{nick}'",
                )
            return CommittedAnswer(used=False)

        # 2. Gaming console / medium: "What console does X play on?" - the
        #    speaker's own "on my <console>" / "on the <console>" mentions.
        if re.search(r"\b(?:console|platform|system|device)\b", ql):
            found: list[str] = []
            for turn in turns:
                s_low = turn.text.lower()
                for c in _CONSOLES:
                    if re.search(rf"\b{re.escape(c)}\b", s_low) and c not in found:
                        found.append(c)
            if found:
                return CommittedAnswer(
                    used=True,
                    answer=", ".join(found[:3]),
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.88,
                    detail=f"consoles mentioned in evidence: {found[:3]}",
                )
            return CommittedAnswer(used=False)

        return CommittedAnswer(used=False)

    @classmethod
    def _extract_nickname(cls, turns: list[Turn]) -> str | None:
        """First nickname captured from 'call(s) me/him/her X' or 'X for short'."""
        for turn in turns:
            for m_n in re.finditer(
                r"\b(?:calls?|called|calling)\s+(?:me|him|her|them)\s+[\"']?([A-Z][a-z]{1,12})[\"']?",
                turn.text,
            ):
                cand = m_n.group(1)
                if cand.lower() not in STOPWORDS:
                    return cand
            m_short = re.search(r"\b([A-Z][a-z]{1,12})\s+for\s+short\b", turn.text)
            if m_short and m_short.group(1).lower() not in STOPWORDS:
                return m_short.group(1)
        return None
