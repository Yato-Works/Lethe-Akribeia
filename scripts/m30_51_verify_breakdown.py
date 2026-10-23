#!/usr/bin/env python3
"""Verify the exact case breakdown: 19 correct, 6 recovered, 8 wrong, 1 unclear."""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
rows = prod["rows"]

baseline_correct = 0
treatment_correct = 0
recovered = 0
regressed = 0
unchanged_correct = 0
unchanged_wrong = 0
changed_no_diff = 0

for r in rows:
    qid = r["qid"]
    gold = r["gold"]
    raw = r.get("predicted", "")
    bl = LoCoMoAdapter.score_binary(r.get("category", 0), gold, raw)

    # Recompute treatment (need to run the actual fallback)
    # Instead, let's just check the ab_results file
    pass

# Check saved results
ab = json.loads((PROJECT_ROOT / "results/m30_51/ab_results_v2.json").read_text(encoding="utf-8"))

print("EXACT CASE BREAKDOWN VERIFICATION")
print("="*50)
print(f"Total: {ab['summary']['total']}")
print(f"Baseline correct: {ab['summary']['baseline_correct']}")
print(f"Treatment correct: {ab['summary']['treatment_correct']}")
print(f"Recovered: {ab['summary']['recovered']}")
print(f"Regressed: {ab['summary']['regressed']}")
print(f"Changed no diff: {ab['summary']['changed_no_diff']}")
print()

# Detailed breakdown
print("DETAILED PER-CASE CLASSIFICATION")
print("="*50)
for r in ab["rows"]:
    status = r["change"]
    bl_c = r["baseline_correct"]
    tr_c = r["treatment_correct"]

    if bl_c and tr_c:
        category = "✅ Already correct (unchanged)"
    elif not bl_c and tr_c:
        category = "✅ RECOVERED"
    elif bl_c and not tr_c:
        category = "❌ REGRESSED"
    elif not bl_c and not tr_c and status == "changed_no_diff":
        category = "⚠️ Changed, still wrong"
    elif not bl_c and not tr_c and status == "same":
        category = "ℹ️ Already wrong, unchanged"
    else:
        category = f"?UNKNOWN: bl={bl_c} tr={tr_c} chg={status}"

    print(f"  {r['qid']:20s} {category}")
    if status == "changed_no_diff" or status == "same":
        if not tr_c:
            print(f"    gold={r['gold'][:40]}")
            print(f"    raw={r['baseline_raw'][:50]} -> trt={r['treatment_raw'][:50]}")

# Count
correct_count = sum(1 for r in ab["rows"] if r["treatment_correct"])
recovered_count = sum(1 for r in ab["rows"] if not r["baseline_correct"] and r["treatment_correct"])
wrong_count = sum(1 for r in ab["rows"] if not r["treatment_correct"])
changed_wrong = sum(1 for r in ab["rows"] if not r["baseline_correct"] and not r["treatment_correct"] and r["change"] == "changed_no_diff")
unchanged_wrong = sum(1 for r in ab["rows"] if not r["baseline_correct"] and not r["treatment_correct"] and r["change"] == "same")

print()
print("="*50)
print(f"Total correct (treatment): {correct_count}")
print(f"Total wrong (treatment): {wrong_count}")
print(f"  - Changed but wrong: {changed_wrong}")
print(f"  - Unchanged (was wrong, still wrong): {unchanged_wrong}")
print(f"Recovered: {recovered_count}")
print(f"Regression: {sum(1 for r in ab['rows'] if r['baseline_correct'] and not r['treatment_correct'])}")
