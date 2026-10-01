"""Unit tests for DerivationScaffolder and MSC compilation with scaffolding."""

from __future__ import annotations

from artificial_memory.context.derivation_scaffold import DerivationScaffolder
from artificial_memory.context.msc_compiler import MinimumSufficientContextCompiler
from artificial_memory.core.ir.memory_types import ApexMemoryUnit, MemoryRole, QueryIntent
from artificial_memory.core.ir.structured import StructuredIR


def test_derivation_scaffolder_temporal_calculation() -> None:
    grounding = "[Temporal Calculation: Event 1 ('visit MoMA') occurred on 2023-01-08. Event 2 ('exhibit') occurred on 2023-01-15. Exactly 7 days passed.]"
    scaffold = DerivationScaffolder.scaffold_temporal(grounding, "How many days passed?")
    assert scaffold is not None
    assert "[DERIVATION SCAFFOLD - TEMPORAL CALCULATION]" in scaffold
    assert "Exactly 7 days passed." in scaffold


def test_derivation_scaffolder_temporal_ordering() -> None:
    grounding = "[Temporal Ordering: 'first event' occurred on 2023-01-01. 'second event' occurred on 2023-01-02. The event that happened first is 'first event'.]"
    scaffold = DerivationScaffolder.scaffold_temporal(grounding, "Which happened first?")
    assert scaffold is not None
    assert "[DERIVATION SCAFFOLD - CHRONOLOGICAL ORDER]" in scaffold
    assert "The event that happened first is 'first event'." in scaffold


def test_derivation_scaffolder_aggregation() -> None:
    records = [
        ApexMemoryUnit(
            ir=StructuredIR(
                entity="clothing",
                property="bought",
                value="2 jackets",
                raw_content="[s1 on 2023-04-01] user: I bought 2 jackets yesterday.",
            ),
            role=MemoryRole.EVIDENCE,
        ),
        ApexMemoryUnit(
            ir=StructuredIR(
                entity="clothing",
                property="bought",
                value="3 shirts",
                raw_content="[s2 on 2023-04-10] user: I ordered 3 shirts online.",
            ),
            role=MemoryRole.EVIDENCE,
        ),
    ]
    scaffold = DerivationScaffolder.scaffold_aggregation(
        "How many items of clothing did I buy?",
        QueryIntent.AGGREGATION_QUERY,
        records,
    )
    assert scaffold is not None
    assert "[DERIVATION SCAFFOLD - MULTI-SESSION AGGREGATION]" in scaffold
    assert "Evidence identified across 2 distinct sessions" in scaffold


def test_msc_compiler_with_derivation_scaffolding() -> None:
    compiler = MinimumSufficientContextCompiler(derivation_scaffolding=True)
    records = [
        StructuredIR(
            entity="Galaxy",
            property="device",
            value="Samsung Galaxy S22 phone",
            time_scope="2023-03-15 10:00",
            raw_content="[s1 on 2023/03/15 (Wed) 10:00] user: I just unboxed the Samsung Galaxy S22 phone!",
        ),
        StructuredIR(
            entity="Dell",
            property="device",
            value="Dell XPS 13 laptop",
            time_scope="2023-03-15 14:30",
            raw_content="[s2 on 2023/03/15 (Wed) 14:30] user: Later that day I received the Dell XPS 13 laptop.",
        ),
    ]
    query = "Which device did I receive first, the Samsung Galaxy S22 or the Dell XPS 13?"
    pcc = compiler.compile(query, records)
    assert pcc.context_text != ""
    assert "[DERIVATION SCAFFOLD - CHRONOLOGICAL ORDER]" in pcc.context_text
    assert "samsung galaxy s22" in pcc.context_text
