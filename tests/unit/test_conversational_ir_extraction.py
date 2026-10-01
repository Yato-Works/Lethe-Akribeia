"""Unit tests for conversational SPO pattern extraction and speaker disambiguation in UniversalIRExtractor."""

from __future__ import annotations

from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.core.ir import IRRelation


def test_conversational_actions_extraction() -> None:
    extractor = UniversalIRExtractor()

    # Pattern H: Action (bought a Sony camera)
    recs = extractor.extract("I bought a Sony camera last week.", default_source="Caroline")
    assert len(recs) == 1
    r = recs[0]
    assert r.entity == "caroline"
    assert r.property == "bought"
    assert "sony camera" in r.value.lower()
    assert r.relation == IRRelation.BEHAVIOR


def test_conversational_preferences_extraction() -> None:
    extractor = UniversalIRExtractor()

    # Pattern I: Preference (loves painting sunrise)
    recs = extractor.extract("I love painting lake sunrises.", default_source="Melanie")
    assert len(recs) == 1
    r = recs[0]
    assert r.entity == "melanie"
    assert r.property == "love" or r.property == "loves"
    assert "painting lake sunrises" in r.value.lower()
    assert r.relation == IRRelation.PREFERS


def test_conversational_possession_and_kinship_extraction() -> None:
    extractor = UniversalIRExtractor()

    # Pattern J: Possession (has a golden retriever)
    recs = extractor.extract("I have a golden retriever.", default_source="Dave")
    assert len(recs) == 1
    r = recs[0]
    assert r.entity == "dave"
    assert r.property == "possesses"
    assert "golden retriever" in r.value.lower()


def test_conversational_occupation_extraction() -> None:
    extractor = UniversalIRExtractor()

    # Pattern K: Occupation (works as a data scientist)
    recs = extractor.extract("I work as a data scientist in Berlin.", default_source="Alice")
    assert len(recs) == 1
    r = recs[0]
    assert r.entity == "alice"
    assert r.property == "occupation"
    assert "data scientist" in r.value.lower()
