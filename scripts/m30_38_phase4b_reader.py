#!/usr/bin/env python3
"""M30-38 Phase 4b: Diagnose WHY Reader fails with evidence present."""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
paired = json.loads((PROJECT_ROOT / "results/m30_38/paired_results.json").read_text(encoding="utf-8"))
frozen = json.loads((PROJECT_ROOT / "results/m30_37/frozen_data.json").read_text(encoding="utf-8"))

conversations = frozen["conversations"]
q_by_id = {q["question_id"]: q for q in frozen["questions"]}

def flatten(conv):
    turns = []
    for key in sorted(conv.keys()):
        if key.endswith("_date_time") or not isinstance(conv[key], list):
            continue
        for turn in conv[key]:
            turns.append({"dia_id": turn["dia_id"], "speaker": turn["speaker"], "text": turn["text"]})
    return turns

print("=" * 78)
print("M30-38 Phase 4b: Why Reader fails with evidence")
print("=" * 78)
print()

reader_fails = [p for p in paired if p["treatment_answer_in_input"] and not p["treatment_correct"]]
print(f"Cases: {len(reader_fails)}")
print()

# Categorize failure mode
modes = {"WRONG_TURN_SELECTED": 0, "WRONG_ENTITY": 0, "ABSTAINED": 0, "OTHER": 0}

for p in reader_fails[:15]:
    q = q_by_id[p["question_id"]]
    turns = flatten(conversations[q["conversation_id"]])
    
    # Count how many turns in input have dates
    input_turns = [t for t in turns if t["dia_id"] in p.get("evidence_ids_in_input", [])]
    
    reader_out = p["treatment_reader"].lower()
    if "don't know" in reader_out or "does not" in reader_out or "cannot" in reader_out:
        modes["ABSTAINED"] += 1
        mode = "ABSTAINED"
    else:
        # Did Reader pick a date that appears in the context?
        reader_date = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}", reader_out)
        if reader_date:
            # Is that date in a turn spoken by the target?
            found = False
            for t in turns:
                if reader_date.group(0).lower() in t["text"].lower():
                    found = True
                    break
            if found:
                modes["WRONG_TURN_SELECTED"] += 1
                mode = "WRONG_TURN_SELECTED (date exists in context but wrong turn)"
            else:
                modes["OTHER"] += 1
                mode = "OTHER"
        else:
            modes["OTHER"] += 1
            mode = "OTHER"
    
    print(f"  {p['question_id']} [{p['category']}]: {mode}")
    print(f"    Expected: {p['expected']}")
    print(f"    Reader: {p['treatment_reader'][:80]}")

print()
print("--- Failure mode counts ---")
for k, v in modes.items():
    print(f"  {k}: {v}")
print()

# How many distinct dates in typical context?
print("--- Context analysis ---")
for p in reader_fails[:5]:
    q = q_by_id[p["question_id"]]
    turns = flatten(conversations[q["conversation_id"]])
    input_ids = p.get("evidence_ids_in_input", [])
    input_turns = [t for t in turns if t["dia_id"] in input_ids]
    dates_in_input = set()
    for t in input_turns:
        for m in re.finditer(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}", t["text"], re.I):
            dates_in_input.add(m.group(0))
    print(f"  {p['question_id']}: {len(input_turns)} turns in input, {len(dates_in_input)} distinct dates")
    if len(dates_in_input) > 1:
        print(f"    AMBIGUOUS CONTEXT: {list(dates_in_input)[:4]}")
