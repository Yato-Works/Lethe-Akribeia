"""Deterministic Memory Engine: Master Orchestrator for Autonomous Memory Subsystems.

Core Philosophy:
"Instead of making the LLM smarter, build a memory engine that answers
 deterministically without needing the LLM to be smart."

Subsystems:
- Subsystem A: CountingEngine (Counting, Cardinality & Set Aggregation)
- Subsystem B: TemporalAlgebraEngine (Allen's Interval Algebra, Weekday Anchor, Duration)
- Subsystem C: RelationalTraverser (SPO Graph, Disjunctions, Preferences, Allergies)
- Subsystem D: EventActionResolver (Actions, Accomplishments, Gifts, Dishes, Themes, Feelings)
- Subsystem E: BooleanVerifier (Polar / Yes-No questions)
- Subsystem F: TemporalAnchorResolver (Date timestamps, Relative days, Event anchors)
- Subsystem G: EntityAttributeResolver (Nicknames, Consoles, Hair color, Lighting, Pets, Tattoos)
"""

from __future__ import annotations

from artificial_memory.skills.answer_committer import (
    CommittedAnswer,
    _is_garbage_answer,
    parse_turns,
)
from artificial_memory.skills.autonomous_engines.arithmetic_difference_engine import (
    ArithmeticDifferenceEngine,
)
from artificial_memory.skills.autonomous_engines.boolean_verifier import BooleanVerifier
from artificial_memory.skills.autonomous_engines.counting_engine import CountingEngine
from artificial_memory.skills.autonomous_engines.entity_attribute_resolver import (
    EntityAttributeResolver,
)
from artificial_memory.skills.autonomous_engines.event_action_resolver import EventActionResolver
from artificial_memory.skills.autonomous_engines.relational_traverser import RelationalTraverser
from artificial_memory.skills.autonomous_engines.temporal_algebra_engine import (
    TemporalAlgebraEngine,
)
from artificial_memory.skills.autonomous_engines.temporal_anchor_resolver import (
    TemporalAnchorResolver,
)


class DeterministicMemoryEngine:
    """Master deterministic reasoning engine spanning Counting, Temporal, Relations, and Events."""

    @classmethod
    def is_counting_q(cls, question: str) -> bool:
        return CountingEngine.is_counting_question(question)

    @classmethod
    def is_duration_q(cls, question: str) -> bool:
        return TemporalAlgebraEngine.is_duration_question(question)

    @classmethod
    def is_date_q(cls, question: str) -> bool:
        return TemporalAlgebraEngine.is_date_question(question)

    @classmethod
    def _is_counting_q(cls, q: str) -> bool:
        return cls.is_counting_q(q)

    @classmethod
    def _is_duration_q(cls, q: str) -> bool:
        return cls.is_duration_q(q)

    @classmethod
    def _is_date_q(cls, q: str) -> bool:
        return cls.is_date_q(q)

    @classmethod
    def resolve(cls, question: str, context: str, category: int | None = None) -> CommittedAnswer:
        """Attempt deterministic resolution through autonomous memory subsystems."""
        turns = parse_turns(context)
        if not turns:
            return CommittedAnswer(used=False, detail="no parsed turns in context")

        # 1. Subsystem H: Arithmetic Difference & Numerical Derivation (Deltas, Savings, Age, Multi-span)
        if ArithmeticDifferenceEngine.is_arithmetic_question(question):
            ans_arith = ArithmeticDifferenceEngine.resolve_arithmetic(question, turns, context)
            if ans_arith.used:
                return ans_arith

        # 2. Subsystem B: Temporal Algebra (Duration / Interval passed)
        if TemporalAlgebraEngine.is_duration_question(question):
            ans = TemporalAlgebraEngine.resolve_duration(question, turns, context)
            if ans.used:
                return ans

        # 3. Subsystem B: Temporal Algebra (Calendar Date & Relative Weekday)
        if TemporalAlgebraEngine.is_date_question(question):
            ans = TemporalAlgebraEngine.resolve_date(question, turns, context)
            if ans.used:
                return ans

        # 3b. Subsystem F: Temporal Anchor Resolver - fills only what B left
        #     open ("yesterday" against the turn's own timestamp).  Running it
        #     after B matters: its relative-day words ("tonight", "today") are
        #     too common to lead a date question.
        if TemporalAnchorResolver.is_temporal_anchor_question(question):
            ans_temp = TemporalAnchorResolver.resolve_temporal_anchor(question, turns, context)
            if ans_temp.used and not _is_garbage_answer(ans_temp.answer):
                return ans_temp

        # 3. Subsystem A: Counting & Cardinality (How many, count, frequency)
        if CountingEngine.is_counting_question(question):
            ans = CountingEngine.resolve_counting(question, turns, context)
            if ans.used:
                return ans

        # 4. Subsystem A: Set Aggregation (What activities/hobbies/books/items?)
        if CountingEngine.is_aggregation_question(question):
            ans = CountingEngine.resolve_aggregation(question, turns, context)
            if ans.used:
                return ans

        # 5. Subsystem D: Event & Action Reasoner (Accomplishments, Foods, Gifts, Themes, Feelings)
        ans_event = EventActionResolver.resolve_event_action(question, turns, context)
        if ans_event.used and not _is_garbage_answer(ans_event.answer):
            return ans_event

        # 6. Subsystem C: Relational Graph Traverser (Favorite, Disjunction, Allergy, Styles)
        ans_rel = RelationalTraverser.resolve_relational(question, turns, context)
        if ans_rel.used and not _is_garbage_answer(ans_rel.answer):
            return ans_rel

        # 7. Subsystem E: Boolean & Polar Question Verifier (Yes/No)
        if BooleanVerifier.is_boolean_question(question):
            ans_bool = BooleanVerifier.resolve_boolean(question, turns, context)
            if ans_bool.used:
                return ans_bool

        # 8. Subsystem G: Entity Attribute Resolver (Nicknames, Consoles, Hair, Lighting, Pets, Books)
        ans_attr = EntityAttributeResolver.resolve_entity_attribute(question, turns, context)
        if ans_attr.used and not _is_garbage_answer(ans_attr.answer):
            return ans_attr

        return CommittedAnswer(used=False, detail="fallback to LLM reader")
