#!/usr/bin/env python3
"""M30-39 Phase 2: Verify the failure taxonomy.

The 27 "Reader failed with evidence in input" cases split into:
- WRONG_TURN_SELECTION: correct date in input, Reader picked another date
- DAY_ARITHMETIC: multi_turn, Reader must compute a day difference
- ANSWER_FORMAT: expected "Session N", Reader gives descriptive text
- ABSTENTION_FORMAT: Reader abstains ("does not provide information") but
  scorer requires the literal substring "I don't know"
- GRANULARITY: expected "M/YYYY", Reader gives a full date

Check each category against the actual outputs.
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
paired = json.loads((PROJECT_ROOT / "results/m30_38/paired_results.json").read_text(encoding="utf-8"))

fails = [p for p in paired if p["treatment_answer_in_input"] and not p["treatment_correct"]]

by_cat = {}
for p in fails:
    by_cat.setdefault(p["category"], []).append(p)

print("=" * 78)
print("M30-39 Phase 2: Failure Taxonomy Verification")
print("=" * 78)
print()

print("Failures by category:")
for cat, items in sorted(by_cat.items()):
    print(f"  {cat}: {len(items)}")
print()

# ---- ABSTENTION_FORMAT check ----
print("--- answerability: did Reader abstain correctly? ---")
for p in by_cat.get("answerability", []):
    reader = p["treatment_reader"].lower()
    abstained = any(s in reader for s in ["don't know", "does not provide", "no information",
                                          "cannot determine", "not specified", "unable to"])
    print(f"  {p['question_id']}: expected={p['expected']!r}")
    print(f"    reader: {p['treatment_reader'][:90]}")
    print(f"    semantically abstained: {abstained}")
print()

# ---- ANSWER_FORMAT check (multi_session) ----
print("--- multi_session: expected 'Session N' vs Reader output ---")
for p in by_cat.get("multi_session", [])[:4]:
    print(f"  {p['question_id']}: expected={p['expected']!r}")
    print(f"    reader: {p['treatment_reader'][:90]}")
print()

# ---- GRANULARITY check ----
print("--- granularity: expected 'M/YYYY' vs Reader output ---")
for p in by_cat.get("granularity", []):
    print(f"  {p['question_id']}: expected={p['expected']!r}")
    print(f"    reader: {p['treatment_reader'][:90]}")
print()

# ---- DAY_ARITHMETIC check ----
print("--- multi_turn: day arithmetic ---")
for p in by_cat.get("multi_turn", [])[:4]:
    print(f"  {p['question_id']}: expected={p['expected']!r}, reader_final={p['treatment_final'][:40]!r}")
print()

# ---- Summary ----
print("=" * 78)
print("TAXONOMY SUMMARY")
print("=" * 78)
tax = {
    "WRONG_TURN_SELECTION": len(by_cat.get("direct_fact", [])) + len(by_cat.get("paraphrase", [])) + len(by_cat.get("entity_disambiguation", [])),
    "DAY_ARITHMETIC": len(by_cat.get("multi_turn", [])),
    "ANSWER_FORMAT_SESSION": len(by_cat.get("multi_session", [])),
    "ABSTENTION_FORMAT": len(by_cat.get("answerability", [])),
    "GRANULARITY": len(by_cat.get("granularity", [])),
}
total = sum(tax.values())
for k, v in tax.items():
    print(f"  {k}: {v}")
print(f"  TOTAL: {total}")
print()

# Which are genuine reasoning failures vs format/scoring?
reasoning = tax["WRONG_TURN_SELECTION"] + tax["DAY_ARITHMETIC"]
format_issue = tax["ANSWER_FORMAT_SESSION"] + tax["ABSTENTION_FORMAT"] + tax["GRANULARITY"]
print(f"Genuine reasoning failures: {reasoning}/{total}")
print(f"Answer-format / scoring mismatches: {format_issue}/{total}")
print()

out = PROJECT_ROOT / "results/m30_39"
out.mkdir(parents=True, exist_ok=True)
(out / "taxonomy.json").write_text(json.dumps({
    "taxonomy": tax, "total": total,
    "genuine_reasoning": reasoning, "format_or_scoring": format_issue,
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'taxonomy.json'}")
