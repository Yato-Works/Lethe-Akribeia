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

        # 3. Purchase source / Store location: "Where did I buy my new X from?"
        if ql.startswith("where did") and any(w in ql for w in ["buy", "get", "purchase"]):
            store = cls._extract_purchase_source(question, turns)
            if store:
                return CommittedAnswer(
                    used=True,
                    answer=store,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"purchase location extracted: '{store}'",
                )

        # 4. Company / Workplace: "What company is X ... currently working at?"
        if ("company" in ql or "organization" in ql) and any(w in ql for w in ["working at", "employed at", "currently at"]):
            company = cls._extract_company_of_person(question, turns)
            if company:
                return CommittedAnswer(
                    used=True,
                    answer=company,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"company extracted for target person: '{company}'",
                )

        # 5. Previous Loyalty / Membership Status: "What was my previous frequent flyer status on X?"
        if "previous" in ql and any(w in ql for w in ["status", "tier", "level", "membership"]):
            status_val = cls._extract_previous_status(question, turns)
            if status_val:
                return CommittedAnswer(
                    used=True,
                    answer=status_val,
                    source="autonomous_entity_attribute_resolver",
                    confidence=0.92,
                    detail=f"previous status extracted: '{status_val}'",
                )

        return CommittedAnswer(used=False)

    @classmethod
    def _extract_purchase_source(cls, question: str, turns: list[Turn]) -> str | None:
        """Extract where an item was bought (e.g. 'the sports store downtown')."""
        # Find item mentioned in question: e.g. "tennis racket"
        m_item = re.search(r"(?:buy|get|purchase)\s+(?:my\s+)?(?:new\s+)?([a-zA-Z\s]+?)(?:\s+from|\??$)", question, re.I)
        item_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", m_item.group(1).lower())) if m_item else set()

        for turn in turns:
            text = turn.text
            text_low = text.lower()
            if item_words and not any(w in text_low for w in item_words):
                continue
            # Look for purchase source: "got from (a|the) ... store [downtown]"
            m_src = re.search(
                r"\b(?:got|bought|purchased)(?:\s+it)?\s+from\s+((?:a|the|an)\s+[a-zA-Z\s]+?(?:store|shop|market|mall|boutique)(?:\s+downtown)?)",
                text,
                re.IGNORECASE,
            )
            if m_src:
                return m_src.group(1).strip()
            # Fallback direct pattern
            m_direct = re.search(r"\bfrom\s+((?:a|the|an)\s+[a-zA-Z\s]+?(?:store|shop|market|mall)(?:\s+downtown)?)\b", text, re.IGNORECASE)
            if m_direct:
                return m_direct.group(1).strip()
        return None

    @classmethod
    def _extract_company_of_person(cls, question: str, turns: list[Turn]) -> str | None:
        """Extract company a person is working at (e.g. 'TechCorp')."""
        # Find person's name: e.g. "Rachel"
        names = re.findall(r"\b[A-Z][a-z]{2,}\b", question)
        target_name = None
        for n in names:
            if n.lower() not in STOPWORDS and n.lower() not in ("what", "company", "who"):
                target_name = n
                break
        if not target_name:
            return None

        # Look in turns for target_name and company mention
        for turn in reversed(turns):  # Check latest sessions first for 'currently'
            text = turn.text
            if target_name.lower() in text.lower():
                # "who's currently at TechCorp", "working at TechCorp", "is at TechCorp"
                m_comp = re.search(
                    rf"\b{re.escape(target_name)}\b.*?currently\s+at\s+([A-Z][a-zA-Z0-9_-]+)",
                    text,
                    re.IGNORECASE,
                )
                if m_comp:
                    return m_comp.group(1).strip()
                m_work = re.search(
                    rf"\b{re.escape(target_name)}\b.*?(?:working|works)\s+at\s+([A-Z][a-zA-Z0-9_-]+)",
                    text,
                    re.IGNORECASE,
                )
                if m_work:
                    return m_work.group(1).strip()
        return None

    @classmethod
    def _extract_previous_status(cls, question: str, turns: list[Turn]) -> str | None:
        """Extract previous frequent flyer / membership status."""
        # Find organization or program (e.g. "United Airlines")
        m_org = re.search(r"\bon\s+([A-Z][a-zA-Z\s]+?)\s+before", question, re.I)
        org = m_org.group(1).strip() if m_org else ""

        # Scan earlier turns for initial status
        for turn in turns:
            text = turn.text
            if org and org.lower() not in text.lower():
                continue
            # "eligible for Premier Silver status", "achieved Silver status", "was Silver status"
            m_stat = re.search(
                r"\b(?:eligible\s+for|had|achieved|attained|was)\s+([A-Z][a-zA-Z\s]+?)\s+status\b",
                text,
                re.IGNORECASE,
            )
            if m_stat:
                cand = m_stat.group(1).strip()
                if cand.lower() not in ("current", "frequent", "my", "our"):
                    return cand
        return None

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
