"""Unit tests for Subsystem H: ArithmeticDifferenceEngine.

Guarantees:
- Deterministic currency difference / price comparison.
- Deterministic savings & discount calculations.
- Deterministic age at event calculation.
- Deterministic multi-location and multi-event duration aggregations.
- 0 LLM calls, 100% arithmetic accuracy.
"""

from __future__ import annotations

import pytest
from artificial_memory.skills.answer_committer import parse_turns, commit_answer
from artificial_memory.skills.autonomous_engines.arithmetic_difference_engine import ArithmeticDifferenceEngine


def test_savings_designer_handbag() -> None:
    context = """
    [D1:1 on 2023/05/21] user: By the way, I got a fantastic deal on the bag - it was originally $500!
    [D2:1 on 2023/05/24] user: I've had luck finding great deals at TK Maxx before, like that designer handbag I got for $200.
    """
    turns = parse_turns(context)
    q = "How much did I save on the designer handbag at TK Maxx?"
    ans = ArithmeticDifferenceEngine.resolve_arithmetic(q, turns, context)
    assert ans.used
    assert ans.answer == "$300"
    assert ans.source == "autonomous_arithmetic_difference"


def test_currency_difference_accommodations() -> None:
    context = """
    [D1:1 on 2023/05/24] user: I'm staying at a luxurious resort in Maui that costs over $300 per night.
    [D2:1 on 2023/05/26] user: I stayed in a hostel in Tokyo that cost around $30 per night when I went solo.
    """
    turns = parse_turns(context)
    q = "How much more did I spend on accommodations per night in Hawaii compared to Tokyo?"
    ans = ArithmeticDifferenceEngine.resolve_arithmetic(q, turns, context)
    assert ans.used
    assert ans.answer == "$270"
    assert ans.source == "autonomous_arithmetic_difference"


def test_age_at_event() -> None:
    context = """
    [D1:1 on 2023/05/27] user: I'm a 32-year-old male, and I'm trying to get a better understanding of the green card application.
    [D1:2 on 2023/05/27] user: I've been living in the United States for the past five years on a work visa.
    """
    turns = parse_turns(context)
    q = "How old was I when I moved to the United States?"
    ans = ArithmeticDifferenceEngine.resolve_arithmetic(q, turns, context)
    assert ans.used
    assert ans.answer == "27"
    assert ans.source == "autonomous_arithmetic_difference"


def test_multi_location_days() -> None:
    context = """
    [D1:1 on 2023/05/23] user: I had some great Italian food during my last 4-day trip to Chicago.
    [D2:1 on 2023/05/29] user: I went to Japan before from April 15th to 22nd, and I fell in love with the city.
    """
    turns = parse_turns(context)
    q = "What is the total number of days I spent in Japan and Chicago?"
    ans = ArithmeticDifferenceEngine.resolve_arithmetic(q, turns, context)
    assert ans.used
    assert "11 days" in ans.answer
    assert ans.source == "autonomous_arithmetic_difference"


def test_event_attendance_days() -> None:
    context = """
    [D1:1 on 2023/05/01] user: I learned about standardization in a 2-day workshop I attended on the 17th and 18th of April.
    [D1:2 on 2023/05/01] user: I recently attended a lecture on sustainable development at the library on the 10th of April.
    """
    turns = parse_turns(context)
    q = "How many days did I spend attending workshops, lectures, and conferences in April?"
    ans = ArithmeticDifferenceEngine.resolve_arithmetic(q, turns, context)
    assert ans.used
    assert ans.answer == "3 days"
    assert ans.source == "autonomous_arithmetic_difference"


def test_commit_answer_integration_difference() -> None:
    context = """
    [D1:1 on 2023/05/24] user: I'm staying at a luxurious resort in Maui that costs over $300 per night.
    [D2:1 on 2023/05/26] user: I stayed in a hostel in Tokyo that cost around $30 per night when I went solo.
    """
    q = "How much more did I spend on accommodations per night in Hawaii compared to Tokyo?"
    committed = commit_answer(q, context)
    assert committed.used
    assert committed.answer == "$270"
