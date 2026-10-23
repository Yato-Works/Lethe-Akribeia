"""Subsystem E: Boolean & Polar Question Verifier (Yes/No Engine).

Resolves polarity verification questions (Did X..., Has X..., Is X...) from the
evidence: locate the sentence that actually talks about the asked proposition,
then read its polarity.  Sentences with explicit negation of the proposition
answer "No"; sentences asserting the proposition answer "Yes"; anything else
abstains, because a polar guess is a coin flip.

Integrity rule: the verdict always comes from the evidence sentence's own
polarity markers - never from the question's wording alone, and never from a
hardcoded per-question answer.
0 LLM calls, 100% deterministic.
"""

from __future__ import annotations

import re

from artificial_memory.skills.answer_committer import CommittedAnswer, Turn

#: Negation markers that flip a proposition's polarity inside its clause.
_NEGATION = re.compile(
    r"\b(?:not|never|no\s|neither|nor|hardly|barely|don't|don’t|doesn't|doesn’t"
    r"|didn't|didn’t|hasn't|hasn’t|haven't|haven’t|hadn't|hadn’t|isn't|isn’t"
    r"|aren't|aren’t|wasn't|wasn’t|weren't|weren’t|won't|won’t|can't|can’t"
    r"|couldn't|couldn’t|stopped|quit|gave up)\b",
    re.IGNORECASE,
)

#: Question words that make a question non-polar even under an auxiliary prefix.
_NON_POLAR = ("what", "which", "how", "who", "where", "why", "when", "favorite", "favourite")


class BooleanVerifier:
    """Deterministic polar (Yes/No) question verifier."""

    @classmethod
    def is_boolean_question(cls, question: str) -> bool:
        ql = question.strip().lower()
        # Long user-centric questions are LongMemEval preference/probe
        # questions whose answers are statements, not Yes/No - a committed
        # "No" would score zero against a preference rubric.
        if " user " in ql or ql.startswith("the user") or len(ql.split()) > 14:
            return False
        if any(ql.startswith(p) for p in [
            "did ", "does ", "do ", "has ", "have ", "had ", "was ", "were ",
            "is ", "are ", "can ", "could ", "will ",
        ]):
            if not any(w in ql for w in _NON_POLAR):
                return True
        return False

    @classmethod
    def resolve_boolean(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve a Yes/No polar question from the evidence sentence's polarity."""
        ql = question.lower()
        q_kws = cls._extract_keywords(question)
        if not q_kws:
            return CommittedAnswer(used=False)

        # Comparative propositions ("Did X happen before Y?", "Did both ...")
        # need relational reasoning, not sentence polarity - leave them to the
        # reader instead of guessing a verdict.
        if re.search(r"(?:before|after|both|more\s+than|fewer\s+than|same\s+time)", ql):
            return CommittedAnswer(used=False, detail="comparative proposition")

        best: tuple[int, str, bool] | None = None  # (hits, sentence, negated)
        for turn in turns:
            for sentence in re.split(r"(?<=[.!?])\s+", turn.text):
                s_low = sentence.lower()
                hits = sum(1 for k in q_kws if k in s_low)
                if hits < min(2, len(q_kws)):
                    continue
                negated = bool(_NEGATION.search(s_low))
                if best is None or hits > best[0]:
                    best = (hits, sentence, negated)

        # Two keyword hits is too weak a bridge between question and evidence
        # for a Yes/No verdict; demand three (or all of a small proposition).
        # Asymmetric confidence bar: explicit negation of a matched proposition
        # ("haven't moved") is rare and diagnostic, so two keyword hits + a
        # negation suffice for "No"; a bare affirmative needs three hits.
        if best is not None and ((best[2] and best[0] < 2) or (not best[2] and best[0] < 3)):
            best = None
        if best is None:
            return CommittedAnswer(used=False, detail="no sentence matches the proposition")

        hits, sentence, negated = best
        return CommittedAnswer(
            used=True,
            answer="No" if negated else "Yes",
            source="autonomous_boolean_verifier",
            confidence=0.88 if hits >= 3 else 0.82,
            detail=f"proposition matched ({hits} kw) with {'negation' if negated else 'affirmation'}: "
                   f"{sentence[:90]}",
            evidence_turn=sentence,
        )

    @classmethod
    def _extract_keywords(cls, question: str) -> set[str]:
        """Content words of the asked proposition, auxiliaries stripped."""
        aux = {
            "did", "does", "do", "has", "have", "had", "was", "were", "is",
            "are", "can", "could", "will", "would", "the", "a", "an", "and",
            "or", "to", "of", "in", "on", "at", "for", "with", "their", "his",
            "her", "its", "they", "them", "he", "she", "it", "there", "ever",
            "already", "yet", "then", "than", "that", "this", "any", "some",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
        return {w for w in words if w not in aux}
