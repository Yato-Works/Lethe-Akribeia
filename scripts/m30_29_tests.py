#!/usr/bin/env python3
"""M30-29 Phase 7: Unit tests for Temporal Resolution Engine."""
import sys, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Import from v2 script
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from m30_29_v2 import (
    resolve_relative,
    detect_relative_expression,
    format_with_granularity,
    detect_question_granularity,
    parse_session_datetime,
    extract_keywords,
)

def test_resolve_relative():
    """Test relative date resolution."""
    tests = [
        # (expr_type, base_date, expected)
        ("next_month", (2024, 1, 6), (2024, 2, None)),
        ("next_month", (2023, 12, 15), (2024, 1, None)),
        ("last_month", (2023, 8, 28), (2023, 7, None)),
        ("last_month", (2023, 1, 15), (2022, 12, None)),
        ("this_month", (2023, 11, 30), (2023, 11, None)),
        ("next_year", (2023, 5, 15), (2024, None, None)),
        ("last_year", (2023, 5, 15), (2022, None, None)),
        ("one_year_ago", (2023, 10, 8), (2022, 10, 8)),  # FIX: preserve month/day
        ("yesterday", (2023, 8, 28), (2023, 8, 27)),
        ("tomorrow", (2023, 8, 28), (2023, 8, 29)),
        ("today", (2023, 8, 28), (2023, 8, 28)),
    ]
    
    passed = 0
    failed = 0
    for expr_type, base, expected in tests:
        result = resolve_relative(expr_type, base)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: {expr_type}({base}) = {result}, expected {expected}")
            failed += 1
    
    print(f"resolve_relative: {passed}/{passed+failed} passed")
    return failed == 0


def test_granularity():
    """Test precision-preserving normalization."""
    tests = [
        # (resolved, granularity, expected)
        ((2022, 10, 8), "month", "October 2022"),
        ((2022, 10, 8), "year", "2022"),
        ((2022, 10, 8), "day", "8 October 2022"),
        ((2023, 9, None), "month", "September 2023"),
        ((2023, None, None), "year", "2023"),
    ]
    
    passed = 0
    failed = 0
    for resolved, granularity, expected in tests:
        result = format_with_granularity(resolved, granularity)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: format_with_granularity({resolved}, {granularity}) = {result}, expected {expected}")
            failed += 1
    
    print(f"format_with_granularity: {passed}/{passed+failed} passed")
    return failed == 0


def test_detect_granularity():
    """Test question granularity detection."""
    tests = [
        # (question, expected_granularity)
        ("When did Melanie get hurt?", "month"),
        ("When is Caroline's youth center putting on a talent show?", "month"),
        ("When did Evan start lifting weights?", "month"),
        ("When will Evan and his partner have their honeymoon in Canada?", "month"),
        ("What year did Melanie graduate?", "year"),
        ("When was Deborah in Bali?", "month"),
    ]
    
    passed = 0
    failed = 0
    for question, expected in tests:
        result = detect_question_granularity(question)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: detect_question_granularity({question!r}) = {result}, expected {expected}")
            failed += 1
    
    print(f"detect_question_granularity: {passed}/{passed+failed} passed")
    return failed == 0


def test_parse_timestamp():
    """Test timestamp parsing."""
    tests = [
        ("1:32 pm on 6 January, 2024", (2024, 1, 6)),
        ("3:19 pm on 28 August, 2023", (2023, 8, 28)),
        ("10:31 am on 13 October, 2023", (2023, 10, 13)),
        ("2023-08-28", (2023, 8, 28)),
        (None, None),
    ]
    
    passed = 0
    failed = 0
    for ts_str, expected in tests:
        result = parse_session_datetime(ts_str)
        if result == expected:
            passed += 1
        else:
            print(f"  FAIL: parse_session_datetime({ts_str!r}) = {result}, expected {expected}")
            failed += 1
    
    print(f"parse_session_datetime: {passed}/{passed+failed} passed")
    return failed == 0


def test_detect_relative_expression():
    """Test relative expression detection."""
    tests = [
        ("Next month.", [("next_month", 0, 10)]),
        ("Last year.", [("last_year", 0, 9)]),
        ("One year ago.", [("one_year_ago", 0, 12), ("years_ago", 0, 12)]),
        ("Yesterday was fun.", [("yesterday", 0, 9)]),
        ("Tomorrow is the big day.", [("tomorrow", 0, 8)]),
        ("No relative expression here.", []),
        ("Two years ago, I moved.", [("years_ago", 0, 12)]),
    ]
    
    passed = 0
    failed = 0
    for text, expected in tests:
        result = detect_relative_expression(text)
        if len(result) == len(expected) and all(r[0] == e[0] for r, e in zip(result, expected)):
            passed += 1
        else:
            print(f"  FAIL: detect_relative_expression({text!r}) = {result}, expected {expected}")
            failed += 1
    
    print(f"detect_relative_expression: {passed}/{passed+failed} passed")
    return failed == 0


def test_extract_keywords():
    """Test keyword extraction."""
    question = "When did Melanie get hurt?"
    kw = extract_keywords(question)
    
    assert "melanie" in kw, f"Expected 'melanie' in {kw}"
    assert "hurt" in kw, f"Expected 'hurt' in {kw}"
    assert "when" not in kw, f"'when' should be stopword"
    assert "did" not in kw, f"'did' should be stopword"
    
    print("extract_keywords: 1/1 passed")
    return True


def test_cross_year_boundary():
    """Test cross-year month resolution."""
    # next month from December -> January next year
    result = resolve_relative("next_month", (2023, 12, 15))
    assert result == (2024, 1, None), f"Expected (2024, 1, None), got {result}"
    
    # last month from January -> December previous year
    result = resolve_relative("last_month", (2023, 1, 15))
    assert result == (2022, 12, None), f"Expected (2022, 12, None), got {result}"
    
    print("cross_year_boundary: 2/2 passed")
    return True


def test_unresolvable():
    """Test that unresolvable expressions don't produce wrong dates."""
    # No base date
    result = resolve_relative("next_month", None)
    assert result is None, f"Expected None, got {result}"
    
    # Unknown expression type
    result = resolve_relative("unknown_type", (2023, 8, 28))
    assert result is None, f"Expected None, got {result}"
    
    print("unresolvable: 2/2 passed")
    return True


def main():
    print("=" * 78)
    print("M30-29 Phase 7: Unit Tests")
    print("=" * 78)
    print()
    
    all_passed = True
    all_passed &= test_resolve_relative()
    all_passed &= test_granularity()
    all_passed &= test_detect_granularity()
    all_passed &= test_parse_timestamp()
    all_passed &= test_detect_relative_expression()
    all_passed &= test_extract_keywords()
    all_passed &= test_cross_year_boundary()
    all_passed &= test_unresolvable()
    
    print()
    print("=" * 78)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 78)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
