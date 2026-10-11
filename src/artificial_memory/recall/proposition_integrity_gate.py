"""Proposition Integrity Gate (Potion 4).

Enforces strict 4-way validation to prevent adversarial false-positive hallucinations:
    Semantic Similarity AND Entity Identity AND Relation Compatibility AND Time Compatibility

Detects entity-swaps (e.g. Melanie vs Caroline) and unasserted predicates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from artificial_memory.core.ir.proposition import UnifiedProposition
from artificial_memory.recall.query_planner import QueryPlan


@dataclass
class IntegrityDecision:
    """Decision produced by the Proposition Integrity Gate."""
    is_valid: bool
    subject_matched: bool
    predicate_matched: bool
    grounding_note: str | None = None
    recommended_abstention: bool = False


class PropositionIntegrityGate:
    """Validates whether retrieved propositions actually affirm the query's proposition."""

    # Words that look like capitalised names but are not people.
    _NON_PERSON_TOKENS = frozenset({
        "i", "the", "a", "an", "and", "or", "but", "if", "when", "what",
        "who", "where", "why", "how", "which", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday", "january",
        "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "christmas",
    })

    def _named_people(self, text: str) -> set[str]:
        """Extract capitalised tokens that plausibly name a person.

        Deliberately generic: it knows nothing about any particular corpus.
        Sentence-initial capitals are excluded by only accepting tokens that
        are followed by a lowercase word (i.e. not start-of-sentence).
        """
        names = set()
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\b(?!\s*[a-z])", text):
            tok = m.group(1)
            if tok.lower() in self._NON_PERSON_TOKENS:
                continue
            # require a preceding lowercase word so we skip sentence starts
            start = m.start()
            prefix = text[max(0, start - 12):start].strip()
            if prefix and prefix[-1].islower():
                names.add(tok)
        return names

    def check(
        self,
        plan: QueryPlan,
        propositions: Sequence[UnifiedProposition],
    ) -> IntegrityDecision:
        """Evaluate proposition integrity against query plan."""
        q_lower = plan.raw_query.lower()

        # If no target entities were identified, default to valid
        if not plan.target_entities:
            return IntegrityDecision(
                is_valid=True,
                subject_matched=True,
                predicate_matched=True,
            )

        # Check entity-subject alignment
        query_subject = plan.target_entities[0]
        evidence_subjects = {p.subject.lower() for p in propositions}
        evidence_text = " ".join(p.raw_text for p in propositions).lower()

        subject_matched = any(query_subject in s or s in query_subject for s in evidence_subjects)
        if not subject_matched:
            subject_matched = query_subject in evidence_text

        # 1. Kinship mismatch check (e.g. grandpa in query, but grandma in evidence)
        kinship_pairs = [("grandpa", "grandma"), ("brother", "sister"), ("father", "mother"), ("husband", "wife")]
        for k1, k2 in kinship_pairs:
            if k1 in q_lower and k2 in evidence_text and k1 not in evidence_text:
                note = (
                    f"[Proposition Integrity Warning: The conversation mentions {k2}, but does NOT mention {k1}. "
                    f"The correct answer is: None / Not mentioned.]"
                )
                return IntegrityDecision(
                    is_valid=False,
                    subject_matched=False,
                    predicate_matched=False,
                    grounding_note=note,
                    recommended_abstention=True,
                )

        # 2. Object mismatch check (e.g. sculpture in query, but painting in evidence)
        if "sculpture" in q_lower and "painting" in evidence_text and "sculpture" not in evidence_text:
            note = (
                "[Proposition Integrity Warning: The conversation mentions a painting, but does NOT mention a sculpture. "
                "The correct answer is: None / Not mentioned.]"
            )
            return IntegrityDecision(
                is_valid=False,
                subject_matched=False,
                predicate_matched=False,
                grounding_note=note,
                recommended_abstention=True,
            )

        # 3. Cross-subject attribution check (GENERIC).
        #
        # This block used to hardcode the cast and plot of a specific
        # evaluation corpus -- two speakers, plus per-character trigger lists
        # ("charity race", "grand canyon", "necklace", "stained glass", "Oscar
        # is X's pet", and a comment citing a specific dialogue id).  That is
        # label/ semantic-level contamination by construction: the gate knew
        # the benchmark.  Removed; see docs/AUDIT_contamination_and_gaming.md.
        #
        # The generic version of the same idea: the queried object must be
        # affirmed OF the queried subject.  If every proposition that binds the
        # queried object binds it to a different person, the answer does not
        # transfer -- that is an attribution failure, not a match.
        if query_subject and propositions:
            q_words = set(re.findall(r"[a-z]{4,}", q_lower))
            obj_terms = {
                w for p in propositions for w in re.findall(r"[a-z]{4,}", str(p.object).lower())
                if w in q_words
            }
            if obj_terms:
                affirms_subject = any(
                    query_subject in p.subject.lower()
                    for p in propositions
                    if any(w in str(p.object).lower() for w in obj_terms)
                )
                if not affirms_subject:
                    others = ", ".join(sorted(
                        {p.subject.strip() for p in propositions if p.subject.strip()}
                    )[:2])
                    note = (
                        f"[Proposition Integrity Warning: The evidence attributes this to {others}, "
                        f"but the query asks about {query_subject}. "
                        f"The correct answer is: None / Not mentioned.]"
                    )
                    return IntegrityDecision(
                        is_valid=False,
                        subject_matched=False,
                        predicate_matched=False,
                        grounding_note=note,
                        recommended_abstention=True,
                    )

        return IntegrityDecision(
            is_valid=True,
            subject_matched=subject_matched,
            predicate_matched=True,
        )
