#!/usr/bin/env python3
"""M30-40 Phase 2-3: Generate Development (40) + Frozen (120) for normalization.

Frozen = 3 groups x 40:
  1. abstention normalization (40)
  2. granularity normalization (40)
  3. negatives / controls (40)

Raw outputs are SYNTHETIC strings that mimic what the Reader would emit,
so we can measure the normalizer in isolation (no LLM re-run, no retrieval).
Expected answers come from structured data, deterministic.

Development uses different surface forms than Frozen.
"""
import json, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
random.seed(31415)

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
MONTH_IDX = {m: i + 1 for i, m in enumerate(MONTHS)}

NAMES = ["Aiden","Bella","Caleb","Diana","Eli","Fiona","Gavin","Hazel",
         "Ivan","Julia","Kevin","Luna","Marcus","Nora","Oscar","Paula"]

# ---------------------------------------------------------------------------
# Abstention surface forms (semantically equivalent to "I don't know.")
# ---------------------------------------------------------------------------
DEV_ABSTENTION = [
    "The context does not provide enough information to answer this question.",
    "There is insufficient information in the provided context.",
    "I cannot determine the answer from the conversation.",
    "The conversation doesn't mention this.",
    "No information about this is available in the context.",
]
FROZEN_ABSTENTION = [
    "The provided context does not contain information about this.",
    "I don't have enough information to answer that.",
    "The evidence does not specify this.",
    "There's no information in the conversation about it.",
    "It is not possible to determine this from the provided context.",
    "The context doesn't include this detail.",
    "The memory does not provide an answer to this question.",
    "The answer is not mentioned in the conversation.",
    "I can't find information about this in the context.",
    "The data provided is insufficient to answer.",
]

# ---------------------------------------------------------------------------
# Negative surface forms (must NOT be converted to abstention)
# ---------------------------------------------------------------------------
DEV_NEGATIVES = [
    "Marcus's favorite hobby is painting. The context does not provide other details.",
    "The answer is July 29, 2025. Note that the context does not mention other dates.",
    "Nora did not attend the event; she was not available that day.",
    "Kevin is not sure, but he thinks it was in 2024. The answer is 2024.",
    "The context mentions several events. The answer is September 3, 2025.",
]
FROZEN_NEGATIVES = [
    "Eli's hobby is photography. The context does not specify how long he has practiced.",
    "The answer is 6/6/2025. The context does not provide additional context.",
    "Diana was not present at the meeting; she joined later.",
    "Gavin is not certain, but the answer is October 20, 2025.",
    "The conversation mentions no other people. The answer is Session 2.",
    "Ivan did not go; instead he stayed home. The answer is that he stayed home.",
    "The context is not clear, but Luna attended on May 4, 2025.",
    "There is no mention of Hazel's favorite color, but her favorite hobby is chess.",
    "The answer is not July but August 12, 2025.",
    "Kevin's answer is that he cannot attend, and he said so on April 9, 2025.",
]


def make_granularity_case(i, group_prefix):
    """Question asks month+year; raw answer gives a full date."""
    name = random.choice(NAMES)
    mi = random.randint(0, 11)
    day = random.randint(1, 28)
    year = random.choice([2024, 2025])
    month = MONTHS[mi]
    raw = f"{name} attended the event on {month} {day}, {year}."
    question = f"In which month and year did {name} attend the event?"
    expected = f"{mi+1}/{year}"
    return {
        "question_id": f"{group_prefix}_g{i:03d}",
        "group": "granularity",
        "question": question,
        "raw_output": raw,
        "expected_answer": expected,
        "should_normalize": True,
        "target": expected,
    }


def make_abstention_case(i, group_prefix, forms):
    name = random.choice(NAMES)
    form = forms[i % len(forms)]
    question = f"What is {name}'s favorite hobby?"
    return {
        "question_id": f"{group_prefix}_a{i:03d}",
        "group": "abstention",
        "question": question,
        "raw_output": form,
        "expected_answer": "I don't know.",
        "should_normalize": True,
        "target": "I don't know.",
    }


def make_negative_case(i, group_prefix, forms):
    """Raw output must remain unchanged; expected is whatever the raw says."""
    form = forms[i % len(forms)]
    name = random.choice(NAMES)
    # question varies: some ask month+year, some ask a fact
    if i % 3 == 0:
        question = f"What is {name}'s hobby?"
        expected = form  # no normalization should apply
    elif i % 3 == 1:
        question = f"When did {name} attend the event?"
        expected = form
    else:
        question = f"On what date did {name} attend the event?"
        expected = form
    return {
        "question_id": f"{group_prefix}_n{i:03d}",
        "group": "negative",
        "question": question,
        "raw_output": form,
        "expected_answer": expected,
        "should_normalize": False,
        "target": form,
    }


def build_development():
    cases = []
    for i in range(15):
        cases.append(make_abstention_case(i, "dev", DEV_ABSTENTION))
    for i in range(15):
        cases.append(make_granularity_case(i, "dev"))
    for i in range(10):
        cases.append(make_negative_case(i, "dev", DEV_NEGATIVES))
    return cases


def build_frozen():
    cases = []
    for i in range(40):
        cases.append(make_abstention_case(i, "frozen", FROZEN_ABSTENTION))
    for i in range(40):
        cases.append(make_granularity_case(i, "frozen"))
    for i in range(40):
        cases.append(make_negative_case(i, "frozen", FROZEN_NEGATIVES))
    return cases


def main():
    out_dir = PROJECT_ROOT / "results/m30_40"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = build_development()
    frozen = build_frozen()

    (out_dir / "dev_cases.json").write_text(
        json.dumps({"cases": dev}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "frozen_cases.json").write_text(
        json.dumps({"cases": frozen}, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    print("Development:")
    for g, n in sorted(Counter(c["group"] for c in dev).items()):
        print(f"  {g}: {n}")
    print(f"  total: {len(dev)}")
    print()
    print("Frozen:")
    for g, n in sorted(Counter(c["group"] for c in frozen).items()):
        print(f"  {g}: {n}")
    print(f"  total: {len(frozen)}")
    print()
    print(f"Saved: {out_dir / 'dev_cases.json'}")
    print(f"Saved: {out_dir / 'frozen_cases.json'}")


if __name__ == "__main__":
    main()
