#!/usr/bin/env python3
"""M30-35 Phase 1: Verify the 19 text fixes."""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
orig = json.loads((PROJECT_ROOT / "results/m30_34/test_data.json").read_text(encoding="utf-8"))
fixed = json.loads((PROJECT_ROOT / "results/m30_34/test_data_fixed.json").read_text(encoding="utf-8"))

orig_conv = orig["conversations"]
fixed_conv = fixed["conversations"]
fixed_q = {q["question_id"]: q for q in fixed["questions"]}

changed_turns = []
for conv_id in orig_conv:
    for key in orig_conv[conv_id]:
        if key.endswith("_date_time") or not isinstance(orig_conv[conv_id][key], list):
            continue
        for i, turn in enumerate(orig_conv[conv_id][key]):
            if turn["text"] != fixed_conv[conv_id][key][i]["text"]:
                changed_turns.append({
                    "conv_id": conv_id,
                    "dia_id": turn["dia_id"],
                    "orig": turn["text"][:80],
                    "fixed": fixed_conv[conv_id][key][i]["text"][:80],
                })

print(f"Total text changes: {len(changed_turns)}")
print()

# Verify: do the fixed texts contain the expected dates?
print("=== Verification: do fixed texts contain expected dates? ===")
passed = 0
failed = 0
for ct in changed_turns:
    for qid, q in fixed_q.items():
        if q["answer_turn"] == ct["dia_id"] and q["conversation_id"] == ct["conv_id"]:
            exp = q["expected_answer"]
            fixed_text = ct["fixed"]
            if exp == "I don't know.":
                break
            m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", exp)
            if m:
                month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                months = ["january","february","march","april","may","june","july","august","september","october","november","december"]
                month_name = months[month - 1].capitalize()
                if month_name in fixed_text and str(day) in fixed_text and str(year) in fixed_text:
                    passed += 1
                else:
                    failed += 1
                    print(f"  FAIL: {qid} expected={exp} fixed={fixed_text[:60]}")
            break

print(f"Passed: {passed}")
print(f"Failed: {failed}")
print()

# Category breakdown
print("=== Category breakdown ===")
from collections import Counter
cat_counts = Counter(q["category"] for q in fixed["questions"])
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat}: {count}")
print(f"  Total: {len(fixed['questions'])}")
print()

# Check: how many questions have timestamps available?
has_ts = 0
no_ts = 0
for q in fixed["questions"]:
    if q["expected_answer"] == "I don't know.":
        continue
    conv = fixed_conv[q["conversation_id"]]
    answer_dia_id = q["answer_turn"]
    found = False
    for key in conv:
        if key.endswith("_date_time"):
            continue
        if not isinstance(conv[key], list):
            continue
        for turn in conv[key]:
            if turn["dia_id"] == answer_dia_id:
                found = True
                break
    if found:
        has_ts += 1
    else:
        no_ts += 1

print(f"Questions with answer turn found: {has_ts}")
print(f"Questions with answer turn NOT found: {no_ts}")
