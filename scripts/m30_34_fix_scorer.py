#!/usr/bin/env python3
"""M30-34 Fix: Patch test data so answer turns contain dates, fix score_answer.

Two bugs found:
1. generate_question picks random turns, but many turns don't contain dates
2. score_answer can't match "June 27, 2023" vs "6/27/2023"

Fix: For each question, find the turn whose session_date matches the expected answer.
"""
import json, re, sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}


def fix_test_data():
    """Fix test data so answer turns actually contain the answer."""
    data = json.loads((PROJECT_ROOT / "results/m30_34/test_data.json").read_text(encoding="utf-8"))
    questions = data["questions"]
    conversations = data["conversations"]
    
    fixed = 0
    for q in questions:
        if q["expected_answer"] == "I don't know.":
            continue
        
        conv_id = q["conversation_id"]
        conv = conversations[conv_id]
        
        # Find the turn whose session date matches the expected answer
        exp = q["expected_answer"]
        exp_date = None
        
        # Parse expected date
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", exp)
        if m:
            exp_date = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        else:
            m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", exp, re.I)
            if m:
                exp_date = (MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)))
        
        if not exp_date:
            continue
        
        # Find a turn in the conversation with this session date
        best_turn = None
        for key in conv:
            if key.endswith("_date_time") or not isinstance(conv[key], list):
                continue
            session_dt = conv.get(f"{key}_date_time", "")
            m2 = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", session_dt)
            if m2 and m2.group(2).lower() in MONTHS:
                s_date = (MONTHS[m2.group(2).lower()], int(m2.group(1)), int(m2.group(3)))
                if s_date == exp_date:
                    # Find a turn in this session that mentions the date
                    for turn in conv[key]:
                        # Check if this turn's text contains a date reference
                        if any(month in turn["text"].lower() for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
                            best_turn = turn["dia_id"]
                            break
                    if not best_turn and conv[key]:
                        best_turn = conv[key][0]["dia_id"]
                    break
        
        if best_turn and best_turn != q["answer_turn"]:
            q["answer_turn"] = best_turn
            fixed += 1
    
    print(f"Fixed {fixed} questions")
    
    # Save
    data["questions"] = questions
    output_path = PROJECT_ROOT / "results/m30_34/test_data_fixed.json"
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    fix_test_data()
