#!/usr/bin/env python3
"""M30-40 Phase 5: Post-hoc safety audit of the normalizer.

The frozen result is FINAL (40 -> 108). This script only DIAGNOSES the
12 non-recovered abstention cases and checks for any meaning change.
It does NOT re-tune and re-score the frozen set as a new result.
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m30_40_normalize import normalize, FACTUAL_OVERRIDE_PATTERNS, ABSTENTION_PATTERNS

res = json.loads((PROJECT_ROOT / "results/m30_40/evaluation_results.json").read_text(encoding="utf-8"))
frozen = res["frozen"]

print("=" * 78)
print("M30-40 Phase 5: Post-hoc Audit (diagnosis only, not a re-score)")
print("=" * 78)
print()

# 1. All cases where treatment is still wrong
still_wrong = [r for r in frozen["rows"] if not r["treatment_correct"]]
print(f"Frozen still-wrong after normalization: {len(still_wrong)}")
from collections import Counter
by_group = Counter(r["group"] for r in still_wrong)
for g, n in sorted(by_group.items()):
    print(f"  {g}: {n}")
print()

# 2. Diagnose the abstention gaps
print("--- Abstention coverage gaps ---")
gaps = {}
for r in still_wrong:
    if r["group"] != "abstention":
        continue
    raw = r["raw"]
    low = raw.lower()
    # which override fired?
    hit_override = [p for p in FACTUAL_OVERRIDE_PATTERNS if re.search(p, low)]
    hit_abst = [p for p in ABSTENTION_PATTERNS if re.search(p, low)]
    key = raw[:50]
    gaps.setdefault(key, {"count": 0, "override": hit_override, "abst": hit_abst})
    gaps[key]["count"] += 1

for text, info in gaps.items():
    print(f"  '{text}...' (x{info['count']})")
    if info["override"]:
        print(f"    BLOCKED by FACTUAL_OVERRIDE: {info['override']}")
    elif not info["abst"]:
        print(f"    NO abstention pattern matches (coverage gap)")
    else:
        print(f"    pattern matched but status not NORMALIZED?")
print()

# 3. Safety: any regression or false positive?
print("--- Safety check ---")
print(f"  Regressed (was correct, now wrong): {frozen['regressed']}")
print(f"  False positives (negative normalized): {frozen['false_positives']}")
print(f"  Meaning changed: {frozen['meaning_changed']}")
print()

# 4. What a corrected pattern set would recover (POST-HOC, not a frozen score)
print("--- POST-HOC: what corrected patterns would recover ---")
print("(This is a diagnosis, NOT a new frozen result.)")
print()

CORRECTED_ABST = [
    r"i\s*(do\s*not|don'?t)\s*have\s+(enough|sufficient)\s+information",
    r"(the\s+)?data\s+(provided\s+)?is\s+(insufficient|not\s+enough)",
    r"(the\s+)?answer\s+is\s+not\s+(mentioned|provided|stated|given|available|in)",
]
# also fix the greedy override: require the 'answer is X' to be followed by
# a NON-negated concrete value
FIXED_OVERRIDE = [
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}/\d{4}\b",
    r"\b\d+\s+days?\b",
    r"\bsession\s+\d+\b",
]

would_recover = 0
for r in still_wrong:
    if r["group"] != "abstention":
        continue
    low = r["raw"].lower()
    blocked = any(re.search(p, low) for p in FIXED_OVERRIDE)
    if blocked:
        continue
    if any(re.search(p, low) for p in CORRECTED_ABST):
        would_recover += 1

print(f"  Additional abstention recoveries with corrected patterns: {would_recover}")
print(f"  Projected frozen (diagnostic only): {frozen['treatment_correct']} + {would_recover} = {frozen['treatment_correct'] + would_recover}/120")
print()
print("  NOTE: These corrected patterns must be validated on a NEW frozen set.")
print("        This 120-case frozen result is final at 108/120.")

out = PROJECT_ROOT / "results/m30_40"
(out / "audit.json").write_text(json.dumps({
    "frozen_final": frozen["treatment_correct"],
    "still_wrong": len(still_wrong),
    "still_wrong_by_group": dict(by_group),
    "gaps": {k: {"count": v["count"], "override": v["override"], "abst": v["abst"]} for k, v in gaps.items()},
    "would_recover_posthoc": would_recover,
    "note": "post-hoc diagnosis, not a frozen score",
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {out / 'audit.json'}")
