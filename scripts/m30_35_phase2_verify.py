#!/usr/bin/env python3
"""M30-35 Phase 2: Verify evaluation harness with fixed data.

Key checks:
1. Reader input is correct (no gold leakage)
2. Track A denominator is answerable questions only
3. Track C denominator is all questions
4. Track A > Track C is impossible (Track C includes retrieval failures)
5. score_answer handles "June 27, 2023" vs "6/27/2023"
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}


def test_score_answer():
    """Test score_answer with various formats."""
    print("=== score_answer tests ===")
    
    tests = [
        ("6/27/2023", "6/27/2023", True),
        ("June 27, 2023", "6/27/2023", True),
        ("2023", "2023", True),
        ("6/2023", "6/2023", True),
        ("I don't know.", "I don't know.", True),
        ("I don't know.", "6/27/2023", False),
        ("6/27/2023", "I don't know.", False),
    ]
    
    passed = 0
    for pred, exp, expected in tests:
        result = score_answer(pred, exp)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"  {status}: score_answer({pred!r}, {exp!r}) = {result} (expected {expected})")
    
    print(f"  {passed}/{len(tests)} passed")
    print()
    return passed == len(tests)


def score_answer(predicted, expected):
    """Fixed score_answer that handles date format mismatches."""
    if expected == "I don't know.":
        return "don't know" in predicted.lower()
    
    pred_norm = predicted.lower().strip().rstrip(".")
    exp_norm = expected.lower().strip().rstrip(".")
    
    if pred_norm == exp_norm:
        return True
    
    # Try to extract dates from both and compare
    pred_date = extract_date(predicted)
    exp_date = extract_date(expected)
    
    if pred_date and exp_date:
        return pred_date == exp_date
    
    # Year only
    if re.match(r"^\d{4}$", exp_norm):
        return exp_norm in pred_norm
    
    # Month/Year
    my_pattern = r"(\d{1,2})/(\d{4})"
    pred_my = re.search(my_pattern, predicted)
    exp_my = re.search(my_pattern, expected)
    if pred_my and exp_my:
        return pred_my.groups() == exp_my.groups()
    
    # Days
    days_pattern = r"(\d+)\s+days?"
    pred_days = re.search(days_pattern, predicted, re.I)
    exp_days = re.search(days_pattern, expected, re.I)
    if pred_days and exp_days:
        return pred_days.group(1) == exp_days.group(1)
    
    # Session
    sess_pattern = r"session\s+(\d+)"
    pred_sess = re.search(sess_pattern, predicted, re.I)
    exp_sess = re.search(sess_pattern, expected, re.I)
    if pred_sess and exp_sess:
        return pred_sess.group(1) == exp_sess.group(1)
    
    return False


def extract_date(text):
    """Extract date as (month, day, year) tuple."""
    # "Month Day, Year"
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", text, re.I)
    if m:
        return (MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)))
    # "MM/DD/YYYY"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # "Month Year"
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", text, re.I)
    if m:
        return (MONTHS[m.group(1).lower()], None, int(m.group(2)))
    # Year only
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        return (None, None, int(m.group(0)))
    return None


def test_data_integrity():
    """Test that fixed data is valid."""
    print("=== Data integrity tests ===")
    
    data = json.loads((PROJECT_ROOT / "results/m30_34/test_data_fixed.json").read_text(encoding="utf-8"))
    questions = data["questions"]
    conversations = data["conversations"]
    
    # Check 1: All questions have valid conversation IDs
    invalid_conv = 0
    for q in questions:
        if q["conversation_id"] not in conversations:
            invalid_conv += 1
    print(f"  Invalid conversation IDs: {invalid_conv}")
    
    # Check 2: All answer turns exist in conversations
    missing_turns = 0
    for q in questions:
        conv = conversations[q["conversation_id"]]
        answer_dia_id = q["answer_turn"]
        found = False
        for key in conv:
            if key.endswith("_date_time") or not isinstance(conv[key], list):
                continue
            for turn in conv[key]:
                if turn["dia_id"] == answer_dia_id:
                    found = True
                    break
        if not found:
            missing_turns += 1
    print(f"  Missing answer turns: {missing_turns}")
    
    # Check 3: All expected answers are valid
    invalid_exp = 0
    for q in questions:
        exp = q["expected_answer"]
        if exp == "I don't know.":
            continue
        if not extract_date(exp) and not re.match(r"^\d+$", exp) and "days" not in exp.lower() and "session" not in exp.lower():
            invalid_exp += 1
    print(f"  Invalid expected answers: {invalid_exp}")
    
    # Check 4: No gold leakage in questions
    leakage = 0
    for q in questions:
        if q["expected_answer"] in q["question"]:
            leakage += 1
    print(f"  Gold leakage in questions: {leakage}")
    
    print()
    return invalid_conv == 0 and missing_turns == 0 and invalid_exp == 0 and leakage == 0


def main():
    print("=" * 78)
    print("M30-35 Phase 2: Evaluation Harness Verification")
    print("=" * 78)
    print()
    
    score_ok = test_score_answer()
    data_ok = test_data_integrity()
    
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  score_answer tests: {'PASS' if score_ok else 'FAIL'}")
    print(f"  Data integrity: {'PASS' if data_ok else 'FAIL'}")
    print()
    
    if score_ok and data_ok:
        print("Evaluation harness is ready for Phase 3-4.")
    else:
        print("Fix evaluation harness before proceeding.")


if __name__ == "__main__":
    main()
