"""Unit tests for SpeakerAttributionNormalizer."""

from __future__ import annotations

from artificial_memory.compiler.speaker_normalizer import SpeakerAttributionNormalizer


def test_normalize_turn_pronoun_disambiguation() -> None:
    normalizer = SpeakerAttributionNormalizer()

    # Case 1: Simple subject and relational disambiguation
    raw = "I love hiking in the mountains and my dog loves running."
    normalized = normalizer.normalize_turn(raw, speaker="Melanie")
    assert "Melanie love hiking" in normalized or "Melanie loves hiking" in normalized
    assert "Melanie's dog loves running" in normalized
    assert "my dog" not in normalized

    # Case 2: Contractions
    raw_contraction = "I'm going to New York next month and I've booked the flight."
    normalized_c = normalizer.normalize_turn(raw_contraction, speaker="Caroline")
    assert "Caroline is going" in normalized_c
    assert "Caroline has booked" in normalized_c

    # Case 3: Reply quote preservation (reply quotes belong to the other speaker)
    reply_raw = '(In reply to Melanie: "Did you buy the tickets?") I bought them yesterday.'
    normalized_reply = normalizer.normalize_turn(reply_raw, speaker="Caroline")
    assert normalized_reply.startswith('(In reply to Melanie: "Did you buy the tickets?")')
    assert "Caroline bought them yesterday" in normalized_reply


def test_generic_speakers_untouched() -> None:
    normalizer = SpeakerAttributionNormalizer()

    # Generic speaker roles should remain untouched
    text = "I am a helpful assistant."
    assert normalizer.normalize_turn(text, speaker="assistant") == text
    assert normalizer.normalize_turn(text, speaker="user") == text


def test_extract_ownership_bindings() -> None:
    normalizer = SpeakerAttributionNormalizer()

    text = "My sister is a violinist and my cat is sleeping."
    bindings = normalizer.extract_ownership_bindings(text, speaker="Caroline")

    assert ("Caroline", "has_relation", "Caroline's sister") in bindings
    assert ("Caroline", "has_relation", "Caroline's cat") in bindings
