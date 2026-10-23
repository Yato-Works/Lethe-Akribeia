"""Self-Anchor & Dual-Entity Expander for Multi-Session Coverage Engine.

Deconstructs pairwise comparative, differential, and dual-entity queries into:
1. Target Entity Requirement (e.g., Alex birth, grandma age, doctor visit, Hawaii)
2. Self-Anchor or Second-Entity Requirement (e.g., my birth year, my age, bed time, NYC)

Prevents the "one-legged retrieval" failure where only one entity is retrieved,
leaving the reader with insufficient evidence to perform comparison or aggregation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PairwiseEvidenceRequirement:
    """Represents a paired evidence requirement for a comparison or difference query."""
    query_type: str                   # 'age_difference', 'relative_schedule', 'dual_entity', 'self_anchor'
    entity_a: str                     # Primary entity (or target)
    entity_b: str                     # Secondary entity (or self-anchor)
    subquery_a: str                   # Targeted subquery to retrieve entity A evidence
    subquery_b: str                   # Targeted subquery to retrieve entity B evidence


class SelfAnchorExpander:
    """Deconstructs queries needing self-reference pairs or dual-entity pairs."""

    # 1. "How old was I when [Entity] was born?"
    _RE_WHEN_BORN = re.compile(
        r"\bhow old was i when\s+([a-zA-Z0-9_\s]+?)\s+(?:was|were)\s+born\b",
        re.I
    )

    # 2. "How many years older/younger is [Entity] than me?"
    _RE_OLDER_YOUNGER_THAN_ME = re.compile(
        r"\bhow many years\s+(?:older|younger)\s+is\s+([a-zA-Z0-9_\s]+?)\s+than\s+(?:me|myself|i)\b",
        re.I
    )

    # 3. "How many years older am I than when I [Event]?"
    _RE_OLDER_THAN_EVENT = re.compile(
        r"\bhow many years\s+older\s+am\s+i\s+than\s+when\s+i\s+(.+?)\??$",
        re.I
    )

    # 4. "What time did I go to bed on the day before [Event]?"
    _RE_DAY_BEFORE = re.compile(
        r"\b(?:what time|when)\s+did i\s+(?:go to bed|sleep|wake up)\s+on the day before\s+(.+?)\??$",
        re.I
    )

    # 5. Dual Entity Comparison: "compared to [EntityB]"
    _RE_COMPARED_TO = re.compile(
        r"(?:on|from|for|in)\s+([^,]+?),\s*(?:as\s+)?compared\s+to\s+(?:on|from|for|in\s+)?(.+?)\??$",
        re.I
    )

    # 6. Dual Entity Conjunction in Aggregation: "in [A] and in [B]", "for [A] and [B]", "from [A] and [B]"
    _RE_DUAL_CONJUNCTION = re.compile(
        r"\b(?:in|traveling in|episodes to from|gifts for|lunch meals i got from|between)\s+([^,]+?)\s+and\s+(?:in\s+)?([^?]+?)\??$",
        re.I
    )

    @classmethod
    def analyze(cls, query: str) -> list[PairwiseEvidenceRequirement]:
        """Detect pairwise comparative intent and generate dual evidence requirements."""
        q_clean = query.strip()
        reqs: list[PairwiseEvidenceRequirement] = []

        # Case 1: "How old was I when [Entity] was born?"
        m1 = cls._RE_WHEN_BORN.search(q_clean)
        if m1:
            target = m1.group(1).strip()
            reqs.append(PairwiseEvidenceRequirement(
                query_type="age_difference",
                entity_a=target,
                entity_b="me",
                subquery_a=f"{target} born birth birthday baby",
                subquery_b="current age my age my birthday born birth year retirement retire 30 32 35 40 25 I was born",
            ))
            return reqs

        # Case 2: "How many years older/younger is [Entity] than me?"
        m2 = cls._RE_OLDER_YOUNGER_THAN_ME.search(q_clean)
        if m2:
            target = m2.group(1).strip()
            reqs.append(PairwiseEvidenceRequirement(
                query_type="age_difference",
                entity_a=target,
                entity_b="me",
                subquery_a=f"{target} age born birthday years old",
                subquery_b="my age current age my birthday I am years old retirement born",
            ))
            return reqs

        # Case 3: "How many years older am I than when I [Event]?" (e.g. graduated from college)
        m3 = cls._RE_OLDER_THAN_EVENT.search(q_clean)
        if m3:
            event = m3.group(1).strip()
            reqs.append(PairwiseEvidenceRequirement(
                query_type="age_difference",
                entity_a=event,
                entity_b="current_age",
                subquery_a=f"{event} year graduated college university",
                subquery_b="my age today currently I am years old birthday",
            ))
            return reqs

        # Case 4: "What time did I go to bed on the day before [Event]?"
        m4 = cls._RE_DAY_BEFORE.search(q_clean)
        if m4:
            event = m4.group(1).strip()
            reqs.append(PairwiseEvidenceRequirement(
                query_type="relative_schedule",
                entity_a=event,
                entity_b="bedtime",
                subquery_a=f"{event} appointment visit scheduled doctor",
                subquery_b="went to bed sleep fell asleep woke up asleep bedtime",
            ))
            return reqs

        # Case 5: Comparison "A, compared to B"
        m5 = cls._RE_COMPARED_TO.search(q_clean)
        if m5:
            ent_a = m5.group(1).strip()
            ent_b = m5.group(2).strip()
            reqs.append(PairwiseEvidenceRequirement(
                query_type="dual_entity",
                entity_a=ent_a,
                entity_b=ent_b,
                subquery_a=ent_a,
                subquery_b=ent_b,
            ))
            return reqs

        # Case 6: Dual Conjunction in Aggregation: "A and B"
        m6 = cls._RE_DUAL_CONJUNCTION.search(q_clean)
        if m6:
            ent_a = m6.group(1).strip()
            ent_b = m6.group(2).strip()
            # Clean quotes if present
            ent_a_clean = ent_a.strip("'\"")
            ent_b_clean = ent_b.strip("'\"")
            reqs.append(PairwiseEvidenceRequirement(
                query_type="dual_entity",
                entity_a=ent_a_clean,
                entity_b=ent_b_clean,
                subquery_a=ent_a_clean,
                subquery_b=ent_b_clean,
            ))
            return reqs

        return reqs
