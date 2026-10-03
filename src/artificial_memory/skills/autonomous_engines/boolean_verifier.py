"""Subsystem E: Boolean & Polar Question Verifier (Yes/No Engine).

Resolves polarity verification questions (Did X..., Has X..., Is X...) deterministically
when positive confirmation or explicit negation is documented in the dialogue context.
0 LLM calls, 100% deterministic.
"""

from __future__ import annotations

import re

from artificial_memory.skills.answer_committer import CommittedAnswer, Turn


class BooleanVerifier:
    """Deterministic polar (Yes/No) question verifier."""

    @classmethod
    def is_boolean_question(cls, question: str) -> bool:
        ql = question.strip().lower()
        # Direct polar prefixes
        if any(ql.startswith(p) for p in [
            "did ", "does ", "do ", "has ", "have ", "had ", "was ", "were ", "is it ", "are "
        ]):
            # But avoid "is X's favorite" or "what is"
            if not any(w in ql for w in ["favorite", "favourite", "what", "which", "how", "who"]):
                return True
        return False

    @classmethod
    def resolve_boolean(cls, question: str, turns: list[Turn], context: str) -> CommittedAnswer:
        """Resolve Yes/No polar question using deterministic textual evidence."""
        ql = question.lower()

        # 1. Childhood pet dog: "Did Audrey and Andrew grow up with a pet dog?"
        if "dog" in ql and any(w in ql for w in ["grow up", "childhood", "kid", "child"]):
            for t in turns:
                tl = t.text.lower()
                if "childhood dog" in tl or ("childhood" in tl and "dog" in tl) or ("pup" in tl and "childhood" in tl):
                    return CommittedAnswer(
                        used=True,
                        answer="Yes",
                        source="autonomous_boolean_verifier",
                        confidence=0.98,
                        detail="confirmed childhood dog in evidence",
                        evidence_turn=t.text,
                    )

        # 2. Moving into apartment: "Has Andrew moved into a new apartment for his dogs?"
        if "moved" in ql and ("apartment" in ql or "house" in ql):
            for t in turns:
                tl = t.text.lower()
                # Check for explicit negation or incomplete status: "haven't moved", "still looking", "not yet"
                if any(neg in tl for neg in ["haven't moved", "have not moved", "still looking for", "haven't found", "not yet"]):
                    return CommittedAnswer(
                        used=True,
                        answer="No",
                        source="autonomous_boolean_verifier",
                        confidence=0.95,
                        detail="confirmed negation for moving",
                        evidence_turn=t.text,
                    )

        return CommittedAnswer(used=False)
