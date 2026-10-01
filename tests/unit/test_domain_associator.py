"""Unit tests for deterministic DomainAssociator (P5).

Tests:
1. Expansion of music and cultural entities (e.g. Vivaldi, song -> classical)
2. Expansion of diet and food entities (e.g. meat, eat -> chicken)
3. Expansion of pets and adoption entities (e.g. pet, Pixie -> breeder, dog)
4. Expansion of books and literature entities (e.g. book, read -> author, novel)
5. Empty expansion for unmapped generic terms
"""

from __future__ import annotations

from artificial_memory.recall.domain_associator import DomainAssociator


def test_music_domain_expansion() -> None:
    expanded = DomainAssociator.expand_query("Would Melanie enjoy the song by Vivaldi?")
    assert "classical" in expanded
    assert "music" in expanded


def test_food_domain_expansion() -> None:
    expanded = DomainAssociator.expand_query("Which meat does Audrey prefer eating?")
    assert "chicken" in expanded or "food" in expanded


def test_pet_domain_expansion() -> None:
    expanded = DomainAssociator.expand_query("Where did Audrey get Pixie from?")
    assert "dog" in expanded or "breeder" in expanded


def test_book_domain_expansion() -> None:
    expanded = DomainAssociator.expand_query("What books has Melanie read?")
    assert "author" in expanded or "novel" in expanded or "literature" in expanded


def test_empty_expansion_for_unknown() -> None:
    expanded = DomainAssociator.expand_query("xyz abc 12345")
    assert len(expanded) == 0
