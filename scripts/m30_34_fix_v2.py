#!/usr/bin/env python3
"""M30-34 Fix v2: Rewrite answer turn texts to explicitly contain the expected date."""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MONTHS = {m.lower(): i for i, m in enumerate([
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december"], start=1)}
MONTH_NAMES = {i: m.capitalize() for m, i in MONTHS.items()}


def fix_answer_turns():
    """Rewrite answer turn texts to include the expected date explicitly."""
    data = json.loads((PROJECT_ROOT / "results/m30_34/test_data.json").read_text(encoding="utf-8"))
    questions = data["questions"]
    conversations = data["conversations"]
    
    fixed = 0
    for q in questions:
        if q["expected_answer"] == "I don't know.":
            continue
        
        exp = q["expected_answer"]
        
        # Parse expected date
        exp_month, exp_day, exp_year = None, None, None
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", exp)
        if m:
            exp_month, exp_day, exp_year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        
        if not exp_month:
            continue
        
        # Find the answer turn and rewrite its text
        conv_id = q["conversation_id"]
        conv = conversations[conv_id]
        answer_dia_id = q["answer_turn"]
        
        for key in conv:
            if key.endswith("_date_time") or not isinstance(conv[key], list):
                continue
            for turn in conv[key]:
                if turn["dia_id"] == answer_dia_id:
                    month_name = MONTH_NAMES.get(exp_month, "")
                    # Rewrite text to include the date
                    turn["text"] = f"On {month_name} {exp_day}, {exp_year}, I went to a city for an event. It was a great experience."
                    fixed += 1
                    break
            break
    
    print(f"Rewrote {fixed} answer turn texts")
    
    # Save
    output_path = PROJECT_ROOT / "results/m30_34/test_data_fixed.json"
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    fix_answer_turns()
