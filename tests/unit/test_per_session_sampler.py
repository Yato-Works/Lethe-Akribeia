import pytest
from artificial_memory_gen.compiler.samplers.per_session_sampler import PerSessionSampler


def test_per_session_sampler_basic():
    sampler = PerSessionSampler(turns_per_session=1)
    query = "How many books did I buy in total?"

    turns = [
        ("s1", "2023/01/01", "user", "I bought 2 books today."),
        ("s1", "2023/01/01", "assistant", "That is great! What titles?"),
        ("s1", "2023/01/01", "user", "One fiction and one science book."),
        ("s2", "2023/02/01", "user", "I went hiking in the mountains."),
        ("s3", "2023/03/01", "user", "Bought 3 more books from the bookstore sale."),
        ("s3", "2023/03/01", "assistant", "Nice haul."),
    ]

    sampled = sampler.sample(query, turns, max_total_tokens=1000)
    assert len(sampled) == 2  # s1 and s3
    sids = [s.session_id for s in sampled]
    assert "s1" in sids
    assert "s3" in sids
    assert "s2" not in sids  # Hiking session not relevant to books

    ctx = sampler.format_as_context(sampled)
    assert "[EXHAUSTIVE MULTI-SESSION EVIDENCE TIMELINE" in ctx
    assert "Event 1" in ctx
    assert "Event 2" in ctx
