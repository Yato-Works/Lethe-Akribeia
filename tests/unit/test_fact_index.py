"""Unit tests for generic FactIndex (Recall & Fact Coverage Layer)."""

import pytest
from artificial_memory_gen.recall.fact_index import FactIndex


def test_fact_index_action_extraction():
    turns = [
        ("s1", "2023-01-01", "Alice", "I went skiing with my friends last weekend."),
        ("s2", "2023-01-02", "Alice", "I signed up for a pottery class yesterday."),
        ("s3", "2023-01-03", "Bob", "I love playing guitar in my free time."),
        ("s4", "2023-01-04", "Alice", "Just having a casual day at home."),
    ]
    idx = FactIndex()
    idx.build(turns)

    # Alice's activities
    results = idx.query("What activities does Alice partake in?", turns)
    turn_texts = [t[3] for t in results]
    assert any("skiing" in t for t in turn_texts)
    assert any("pottery" in t for t in turn_texts)
    assert not any("playing guitar" in t for t in turn_texts)  # Bob's activity


def test_fact_index_preference_extraction():
    turns = [
        ("s1", "2023-01-01", "Bob", "My kids are stoked for the robotics competition!"),
        ("s2", "2023-01-02", "Bob", "They love learning about marine biology."),
        ("s3", "2023-01-03", "Bob", "Thanks, that was a great conversation."),
    ]
    idx = FactIndex()
    idx.build(turns)

    results = idx.query("What do Bob's kids like?", turns)
    turn_texts = [t[3] for t in results]
    assert any("robotics" in t for t in turn_texts)
    assert any("marine biology" in t for t in turn_texts)


def test_fact_index_multihop_coreference():
    turns = [
        ("s1", "2023-01-01", "Alice", "I moved from my home country 5 years ago."),
        ("s2", "2023-01-02", "Alice", "This bracelet was a gift from my grandma in my home country, Canada."),
        ("s3", "2023-01-03", "Bob", "That sounds wonderful, Alice."),
    ]
    idx = FactIndex()
    idx.build(turns)

    results = idx.query("Where did Alice move from 5 years ago?", turns)
    turn_texts = [t[3] for t in results]
    assert any("5 years ago" in t for t in turn_texts)
    assert any("Canada" in t for t in turn_texts)
