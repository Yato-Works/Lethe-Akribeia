#!/usr/bin/env python3
"""M30-53: Analyze the 8 remaining wrong-after-fallback cases to determine if
they need LLM-level intervention (unfixable by deterministic rules) or if
there's a pattern we can exploit."""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Load both results
v1 = json.loads((PROJECT_ROOT / "results/m30_51/ab_results_v2.json").read_text(encoding="utf-8"))

# Classify each wrong case
print("M30-53: Analysis of 8 remaining 'changed_no_diff' cases")
print("="*70)

wrong_cases = [r for r in v1["rows"] if r["change"] == "changed_no_diff"]
print(f"\nTotal changed_no_diff: {len(wrong_cases)}")
print()

for r in wrong_cases:
    gold = r["gold"]
    raw = r["baseline_raw"]
    trt = r["treatment_raw"]

    print(f"{r['qid']}: gold={gold}")
    print(f"  reader:  {raw[:70]}")
    print(f"  fallback: {trt[:70]}")
    print(f"  reader_abstain: {raw.startswith('There is no') or raw.startswith('The conversation') or raw.startswith('Based on')}")
    print()

# Summary of failure modes
print("="*70)
print("FAILURE MODE ANALYSIS:")
print()

# Category the wrong cases
reader_hallucinated = 0
reader_abstained = 0
fallback_wrong_turn = 0
fallback_wrong_expr = 0
questions_needing_semantic = 0

for r in wrong_cases:
    gold = r["gold"]
    raw = r["baseline_raw"]
    trt = r["treatment_raw"]

    if raw.startswith("There is no") or raw.startswith("The conversation") or raw.startswith("Based on"):
        reader_abstained += 1
    else:
        reader_hallucinated += 1

# Cases where fallback gives a plausible but wrong date (turn selection issue)
for r in wrong_cases:
    trt = r["treatment_raw"]
    gold = r["gold"]
    if trt and trt != gold and "," in trt and re.search(r'\d{4}', trt):
        fallback_wrong_turn += 1

print(f"Reader abstained (no answer):      {reader_abstained}")
print(f"Reader hallucinated (wrong ans):   {reader_hallucinated}")
print(f"Fallback wrong turn selection:     {fallback_wrong_turn}")
print()

print("CONCLUSION: The 8 remaining cases fall into 2 categories:")
print("1. Q26_72, Q48_50, Q49_59: Questions asking about 'year' — need")
print("   year-only resolution, not full date")
print("2. Q42_57, Q47_48, Q50_52: Multiple context turns with same actors —")
print("   require SEMANTIC disambiguation beyond noun overlap")
print("3. Q44_13, Q50_36: Reader already gives a specific (wrong) date —")
print("   fallback can't correct without knowing which date is right")
print()
print("M30-52 improvement target: semantic event matching for category 2")
