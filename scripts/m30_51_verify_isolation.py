#!/usr/bin/env python3
"""Verify holdout isolation for M30-51 results."""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load production baseline
prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
prod_qids = set(r["qid"] for r in prod["rows"])

# Load ab_results
ab = json.loads((PROJECT_ROOT / "results/m30_51/ab_results_v2.json").read_text(encoding="utf-8"))
ab_qids = set(r["qid"] for r in ab["rows"])

# Check for holdout leakage
extra = ab_qids - prod_qids
missing = prod_qids - ab_qids

print("HOLDOUT ISOLATION VERIFICATION")
print("="*50)
print(f"Production QIDs: {len(prod_qids)}")
print(f"AB result QIDs: {len(ab_qids)}")
print(f"Extra QIDs (not in production = potential holdout leak): {len(extra)}")
print(f"Missing QIDs (in production but not in ab): {len(missing)}")

if not extra and not missing:
    print("✅ PERFECT MATCH — no holdout leakage")
    print(f"   All {len(ab_qids)} QIDs match production exactly")
else:
    if extra:
        print(f"   ❌ LEAKED: {sorted(extra)}")
    if missing:
        print(f"   ❌ MISSING: {sorted(missing)}")

# Also verify all QIDs use Bucket A conv IDs
conv_nums = set()
for qid in ab_qids:
    m = re.match(r"lc_conv-(\d+)_", qid)
    if m:
        conv_nums.add(int(m.group(1)))
        
print(f"\nConv IDs in ab_results: {sorted(conv_nums)}")
print(f"All from Bucket A: {conv_nums == {26, 30, 41, 42, 44, 47, 48, 49, 50}}")
