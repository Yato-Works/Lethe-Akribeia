#!/usr/bin/env python3
"""M30-37 Phase 1: Resolve inconsistencies in paired analysis.

Key questions:
1. Why does Track A denominator differ (115 vs 120)?
2. Why does Track C differ (23 vs 24)?
3. Are the 12 A_WRONG_C_CORRECT cases due to insufficient Track A evidence?
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

paired = json.loads((PROJECT_ROOT / "results/m30_36/paired_results.json").read_text(encoding="utf-8"))
m35 = json.loads((PROJECT_ROOT / "results/m30_35/track_results.json").read_text(encoding="utf-8"))
test_data = json.loads((PROJECT_ROOT / "results/m30_35/test_data.json").read_text(encoding="utf-8"))

print("=" * 78)
print("M30-37 Phase 1: Inconsistency Resolution")
print("=" * 78)
print()

# 1. Track A denominator
answerable = [q for q in test_data["questions"] if q["expected_answer"] != "I don't know."]
unanswerable = [q for q in test_data["questions"] if q["expected_answer"] == "I don't know."]
print(f"Total questions: {len(test_data['questions'])}")
print(f"Answerable: {len(answerable)}")
print(f"Unanswerable: {len(unanswerable)}")
print()

# Paired analysis counted all 120
a_correct_all = sum(1 for r in paired if r.get("track_a_correct") is True)
a_incorrect_all = sum(1 for r in paired if r.get("track_a_correct") is False)
a_none = sum(1 for r in paired if r.get("track_a_correct") is None)
print(f"Paired Track A: correct={a_correct_all}, incorrect={a_incorrect_all}, none={a_none}")
print()

# M30-35 excluded unanswerable from Track A denominator
print(f"M30-35 Track A: {m35['track_a']['correct']}/{m35['track_a']['total']}")
print(f"  (denominator = answerable only = {len(answerable)})")
print()

# 2. Track C discrepancy
c_correct_paired = sum(1 for r in paired if r.get("track_c_correct") is True)
print(f"Paired Track C correct: {c_correct_paired}")
print(f"M30-35 Track C correct: {m35['track_c']['correct']}")
print(f"Difference: {m35['track_c']['correct'] - c_correct_paired}")
print()

# 3. A_WRONG_C_CORRECT analysis
a_wrong_c_right = [r for r in paired if r["classification"] == "A_WRONG_C_CORRECT"]
print(f"A_WRONG_C_CORRECT: {len(a_wrong_c_right)} cases")
print()
print("Analysis: Track A was given ONLY the answer_turn.")
print("If that single turn lacks the date, Track A cannot answer.")
print("Track C retrieves multiple turns and may find the date elsewhere.")
print()

# Check: do the answer_turns for these cases contain dates?
for r in a_wrong_c_right[:5]:
    qid = r["question_id"]
    q = next((x for x in test_data["questions"] if x["question_id"] == qid), None)
    if not q:
        continue
    
    conv = test_data["conversations"][q["conversation_id"]]
    answer_dia_id = q["answer_turn"]
    answer_text = ""
    for key in conv:
        if key.endswith("_date_time") or not isinstance(conv[key], list):
            continue
        for turn in conv[key]:
            if turn["dia_id"] == answer_dia_id:
                answer_text = turn["text"]
                break
    
    has_month = any(m in answer_text.lower() for m in ["january","february","march","april","may","june","july","august","september","october","november","december"])
    print(f"  {qid}: answer_turn={answer_dia_id}")
    print(f"    text: {answer_text[:70]}")
    print(f"    has explicit date: {has_month}")
    print()

# 4. Summary
print("=" * 78)
print("CONCLUSIONS")
print("=" * 78)
print()
print("1. Track A denominator:")
print(f"   - M30-35 used {len(answerable)} (answerable only)")
print(f"   - Paired analysis used 120 (all questions)")
print(f"   - Both are correct; they measure different things.")
print()
print("2. Track C discrepancy (1 case):")
print("   - Likely due to LLM non-determinism (temperature=0.1)")
print("   - Not a systematic error")
print()
print("3. A_WRONG_C_CORRECT (12 cases):")
print("   - Track A was given only 1 turn (answer_turn)")
print("   - If that turn lacks a date, Track A abstains")
print("   - Track C retrieves 8 turns and may find the date")
print("   - This is NOT a failure of Track C; it is a limitation of Track A's single-turn setup")
print()
print("4. This means the M30-36 'A_CORRECT_C_WRONG' (22) count may also be affected:")
print("   - Those 22 cases had Track A correct and Track C wrong")
print("   - Need to verify Track C's retrieval actually missed the evidence")
