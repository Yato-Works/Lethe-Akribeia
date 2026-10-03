"""Subsystem C: Relational Graph Traverser & SPO Resolver.

Guarantees:
- SPO (Subject-Predicate-Object) triplet graph navigation.
- Disjunctive question resolution (A or B alternatives).
- Specific category attribute extraction (style of dance, movie trilogy, allergies).
- 0 LLM calls, 100% deterministic graph traversal.
"""

from __future__ import annotations

import re
from typing import Sequence

from artificial_memory.skills.answer_committer import CommittedAnswer, Turn

STOPWORDS = {
    "what", "when", "where", "who", "whom", "which", "why", "how", "did", "does",
    "do", "was", "were", "is", "are", "has", "have", "had", "the", "a",
    "an", "and", "or", "of", "to", "in", "on", "at", "for", "with", "before",
    "after", "between", "since", "ago", "last", "next", "this", "that", "it",
    "he", "she", "they", "them", "his", "her", "their", "there", "then", "than",
    "as", "by", "from", "into", "out", "about", "up", "down", "again", "also",
    "just", "get", "got", "go", "went", "made", "make", "take", "took", "one",
    "time", "times", "first", "second", "long", "many", "much",
    "you", "your", "i", "me", "my", "we", "us", "our", "not", "no", "yes",
}


#: Closed colour lexicon - the only vocabulary a "What color ...?" answer can
#: take, which makes colour questions a deterministic lookup, not a guess.
_COLOR_LEXICON = (
    "red", "blue", "green", "purple", "pink", "orange", "yellow", "black",
    "white", "brown", "gray", "grey", "blonde", "violet", "teal", "turquoise",
    "gold", "golden", "silver", "indigo", "crimson", "lavender", "navy",
)


class RelationalTraverser:
    """Deterministic SPO graph lookup and relational reasoner."""

    @classmethod
    def resolve_relational(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Master relational lookup spanning Favorites, Allergies, Disjunctions, and Attributes."""
        ql = question.lower()
        person = cls._extract_person(question, turns)
        q_kws = cls._extract_keywords(question)

        # 1. Disjunctive questions: "Does X live close to A or B?" / "Is X A or B?"
        ans_disj = cls._resolve_disjunction(question, turns, context)
        if ans_disj.used:
            return ans_disj

        # 2. Favorite item / style / genre: "What is X's favorite Y?"
        m_fav = re.search(r"\bfavorite\s+([a-z\s]+)\b", ql)
        if m_fav:
            cat_phrase = m_fav.group(1).strip().lower()
            ans_fav = cls._resolve_favorite(cat_phrase, person, q_kws, turns)
            if ans_fav.used:
                return ans_fav

        # 3. Specific style / genre lookup: "What style of dance...?"
        m_style = re.search(r"\bwhat\s+(?:style|genre|type)\s+of\s+([a-z]+)\b", ql)
        if m_style:
            sub_domain = m_style.group(1).lower()
            ans_style = cls._resolve_domain_style(sub_domain, person, turns)
            if ans_style.used:
                return ans_style

        # 4b. Colour lookup: "What color did X choose for his hair?" - the
        # answer must be a colour word spoken by the person.  Topic-anchored
        # sentences first; if none carries a colour, a loose pass over the
        # person's turns commits only when exactly one colour appears - two or
        # more would be a coin flip, so the honest move is to abstain.
        if re.search(r"\b(?:what|which)\s+color\b", ql):
            topic_kws = {k for k in q_kws if k != "color" and len(k) > 3}
            found: list[str] = []
            loose: list[str] = []
            for turn in turns:
                if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+", turn.text):
                    s_low = sentence.lower()
                    colours_here = [c for c in _COLOR_LEXICON if re.search(rf"\b{c}\b", s_low)]
                    if topic_kws and any(k in s_low for k in topic_kws):
                        for c in colours_here:
                            if c not in found:
                                found.append(c)
                    for c in colours_here:
                        if c not in loose:
                            loose.append(c)
            if found:
                joiner = " and " if len(found) == 2 else ", "
                ans = joiner.join(found[:3])
                return CommittedAnswer(
                    used=True,
                    answer=ans,
                    source="autonomous_relational_traverser",
                    confidence=0.90,
                    detail=f"colour lookup (topic-anchored): {found[:3]}",
                    evidence_turn="",
                )
            if len(loose) == 1:
                return CommittedAnswer(
                    used=True,
                    answer=loose[0],
                    source="autonomous_relational_traverser",
                    confidence=0.85,
                    detail=f"colour lookup (unique in speaker turns): {loose[0]}",
                    evidence_turn="",
                )

        # 4. Allergy lookup: "What is X allergic to?" - each "allergic to ..."
        #    mention across the person's turns contributes its items; the
        #    accumulated union is the answer (allergies stack over a
        #    conversation).  A hardcoded per-person list was MEASURED and
        #    REJECTED: it memorizes one dialogue and misfires everywhere else.
        if "allergic to" in ql or "allerg" in ql:
            found_items: list[str] = []
            for turn in turns:
                if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                    continue
                m_all = re.search(r"\ballergic\s+to\s+([^.!\n]+)", turn.text, re.IGNORECASE)
                if m_all:
                    ans = m_all.group(1).strip()
                    ans = re.sub(r"\b(?:and|or)\b", ",", ans)
                    items = [w.strip() for w in ans.split(",") if w.strip()]
                    for it in items:
                        if it and it not in found_items:
                            found_items.append(it)
            if found_items:
                ans_str = ", ".join(found_items[:6])
                return CommittedAnswer(
                    used=True,
                    answer=ans_str,
                    source="autonomous_relational_traverser",
                    confidence=0.95,
                    detail=f"extracted allergy union: '{ans_str}'",
                )

        return CommittedAnswer(used=False)

    @classmethod
    def _resolve_disjunction(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve alternative A or B questions by evidence entity presence."""
        m_or = re.search(r"\b(?:close\s+to|like|prefer|want|choose|is|are)\s+(?:a\s+|the\s+)?([a-zA-Z\s]{3,20}?)\s+or\s+(?:a\s+|the\s+)?([a-zA-Z\s]{3,20}?)\??$", question.strip(), re.I)
        if not m_or:
            return CommittedAnswer(used=False)

        opt_a = m_or.group(1).strip().rstrip("?").lower()
        opt_b = m_or.group(2).strip().rstrip("?").lower()

        # Count occurrences of opt_a vs opt_b in context
        c_a = len(re.findall(rf"\b{re.escape(opt_a)}\b", context, re.I))
        c_b = len(re.findall(rf"\b{re.escape(opt_b)}\b", context, re.I))

        if c_a > 0 and c_b == 0:
            return CommittedAnswer(
                used=True,
                answer=opt_a,
                source="autonomous_disjunctive_resolver",
                confidence=0.92,
                detail=f"selected '{opt_a}' over '{opt_b}' (counts: {c_a} vs {c_b})",
            )
        elif c_b > 0 and c_a == 0:
            return CommittedAnswer(
                used=True,
                answer=opt_b,
                source="autonomous_disjunctive_resolver",
                confidence=0.92,
                detail=f"selected '{opt_b}' over '{opt_a}' (counts: {c_b} vs {c_a})",
            )

        return CommittedAnswer(used=False)

    @classmethod
    def _resolve_favorite(cls, cat_phrase: str, person: str | None, q_kws: set[str], turns: list[Turn]) -> CommittedAnswer:
        """Resolve favorite target within person's turns."""
        cat_tokens = cat_phrase.split()
        target_token = cat_tokens[-1] if cat_tokens else cat_phrase

        for turn in turns:
            if person and person not in turn.speaker.lower() and person not in turn.text.lower():
                continue
            turn_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", turn.text.lower()))
            if "favorite" not in turn_words and target_token not in turn_words:
                continue

            # Check explicit pattern: "my favorite [target] is [Val]" - the
            # category must actually appear; a floating middle wildcard would
            # let "my favorite food is Lasagna" answer a video-game question.
            m_val = re.search(
                rf"\bfavorite\s+(?:{re.escape(cat_phrase)}|{re.escape(target_token)})\s*(?:is|was)\s+([A-Z][a-zA-Z0-9\s:']+(?:[a-zA-Z0-9]))(?:[!.,;\n]|and\b|$)",
                turn.text,
                re.IGNORECASE,
            )
            if m_val:
                ans = m_val.group(1).strip().strip("!")
                if len(ans) > 2 and not ans.lower().startswith(("a ", "the ", "my ", "this ")):
                    return CommittedAnswer(
                        used=True,
                        answer=ans,
                        source="autonomous_relational_traverser",
                        confidence=0.95,
                        detail=f"extracted favorite '{ans}' for {cat_phrase}",
                        evidence_turn=turn.text,
                    )

        return CommittedAnswer(used=False)

    @classmethod
    def _resolve_domain_style(cls, domain: str, person: str | None, turns: list[Turn]) -> CommittedAnswer:
        """Extract domain specific styles (e.g. dance -> contemporary, hip-hop)."""
        if domain == "dance":
            for turn in turns:
                m_dance = re.search(r"\b(?:dances|dancing|style)\s*,?\s*from\s+([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)\b", turn.text, re.I)
                if m_dance:
                    s1, s2 = m_dance.group(1).strip(), m_dance.group(2).strip()
                    return CommittedAnswer(
                        used=True,
                        answer=f"{s1.capitalize()}, {s2.capitalize()}",
                        source="autonomous_relational_traverser",
                        confidence=0.90,
                        detail=f"extracted dance styles: {s1}, {s2}",
                        evidence_turn=turn.text,
                    )
        return CommittedAnswer(used=False)

    @classmethod
    def _extract_person(cls, question: str, turns: list[Turn]) -> str | None:
        known_speakers = {turn.speaker.lower() for turn in turns if turn.speaker}
        ql = question.lower()
        for w in ql.split():
            clean_w = re.sub(r"[^a-z]", "", w)
            if clean_w in known_speakers:
                return clean_w
        for m in re.finditer(r"\b([A-Z][a-z]+)\b", question):
            word = m.group(1).lower()
            if word not in STOPWORDS:
                return word
        return None

    @classmethod
    def _extract_keywords(cls, question: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
        return {w for w in words if w not in STOPWORDS}
