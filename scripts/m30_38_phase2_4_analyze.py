#!/usr/bin/env python3
"""M30-38 Phase 2-3: Analyze recovery/regression causes + retrieval metrics."""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

paired = json.loads((PROJECT_ROOT / "results/m30_38/paired_results.json").read_text(encoding="utf-8"))

print("=" * 78)
print("M30-38 Phase 2: Recovery/Regression Cause Analysis")
print("=" * 78)
print()

recovered = [p for p in paired if p["classification"] == "RECOVERED"]
regressed = [p for p in paired if p["classification"] == "REGRESSED"]

print(f"RECOVERED: {len(recovered)}")
print(f"REGRESSED: {len(regressed)}")
print()

print("--- RECOVERED cases ---")
for p in recovered:
    print(f"{p['question_id']} [{p['category']}]")
    print(f"  Expected: {p['expected']}")
    print(f"  Baseline: rank={p['baseline_rank']}, in_input={p['baseline_answer_in_input']}, final={p['baseline_final'][:50]}")
    print(f"  Treatment: rank={p['treatment_rank']}, in_input={p['treatment_answer_in_input']}, final={p['treatment_final'][:50]}")
    print()

# Classify recovery cause
print("--- Recovery cause classification ---")
cause_counts = {"RETRIEVAL_RECOVERY": 0, "EXTRACTION_RECOVERY": 0, "READER_RECOVERY": 0,
                "ANSWER_EXTRACTION_RECOVERY": 0, "SCORING_EFFECT": 0, "MULTIPLE_STAGES": 0, "UNVERIFIED": 0}

for p in recovered:
    base_in = p["baseline_answer_in_input"]
    treat_in = p["treatment_answer_in_input"]
    
    if not base_in and treat_in:
        cause_counts["RETRIEVAL_RECOVERY"] += 1
        cause = "RETRIEVAL_RECOVERY"
    elif base_in and not treat_in:
        cause = "UNVERIFIED (regression in evidence)"
        cause_counts["UNVERIFIED"] += 1
    elif base_in and treat_in:
        cause = "READER_RECOVERY (evidence present in both, Reader changed)"
        cause_counts["READER_RECOVERY"] += 1
    else:
        cause = "UNVERIFIED (evidence absent in both)"
        cause_counts["UNVERIFIED"] += 1
    
    print(f"  {p['question_id']}: {cause}")

print()
for k, v in cause_counts.items():
    if v > 0:
        print(f"  {k}: {v}")
print()

# =============================================================================
# Phase 3: Retrieval metrics
# =============================================================================
print("=" * 78)
print("M30-38 Phase 3: Retrieval Metrics Verification")
print("=" * 78)
print()

for label, key in [("Baseline", "baseline"), ("Treatment", "treatment")]:
    ranks = [p[f"{key}_rank"] for p in paired]
    r1 = sum(1 for r in ranks if r == 1)
    r5 = sum(1 for r in ranks if r and r <= 5)
    r10 = sum(1 for r in ranks if r and r <= 10)
    retrieved = sum(1 for r in ranks if r is not None)
    in_input = sum(1 for p in paired if p[f"{key}_answer_in_input"])
    
    print(f"{label}:")
    print(f"  Recall@1: {r1}/120")
    print(f"  Recall@5: {r5}/120")
    print(f"  Recall@10: {r10}/120")
    print(f"  Retrieved (any rank): {retrieved}/120")
    print(f"  In Reader input (top-8): {in_input}/120")
    print()

# Key: does retrieval improvement translate to input?
print("--- Retrieval -> Input conversion ---")
base_retrieved = sum(1 for p in paired if p["baseline_rank"] is not None)
treat_retrieved = sum(1 for p in paired if p["treatment_rank"] is not None)
base_input = sum(1 for p in paired if p["baseline_answer_in_input"])
treat_input = sum(1 for p in paired if p["treatment_answer_in_input"])
print(f"Baseline: retrieved={base_retrieved}, in_input={base_input}, conversion={100*base_input/max(base_retrieved,1):.1f}%")
print(f"Treatment: retrieved={treat_retrieved}, in_input={treat_input}, conversion={100*treat_input/max(treat_retrieved,1):.1f}%")
print()

# =============================================================================
# Phase 4: Reader diagnosis
# =============================================================================
print("=" * 78)
print("M30-38 Phase 4: Reader Evidence Utilization")
print("=" * 78)
print()

# Cases where evidence WAS in input but Reader failed
reader_fails = [p for p in paired if p["treatment_answer_in_input"] and not p["treatment_correct"]]
print(f"Treatment: evidence in input but wrong: {len(reader_fails)}")
print()

# Cases where evidence was NOT in input and Reader failed
no_evidence = [p for p in paired if not p["treatment_answer_in_input"] and not p["treatment_correct"]]
print(f"Treatment: evidence NOT in input and wrong: {len(no_evidence)}")
print()

# Cases where evidence in input and correct
reader_ok = [p for p in paired if p["treatment_answer_in_input"] and p["treatment_correct"]]
print(f"Treatment: evidence in input and correct: {len(reader_ok)}")
print()

# Diagnosis
if len(reader_fails) > 0:
    total_with_evidence = len(reader_fails) + len(reader_ok)
    print(f"READER_CAN_ANSWER_WITH_SUPPORT: {len(reader_ok)}/{total_with_evidence} ({100*len(reader_ok)/total_with_evidence:.1f}%)")
    print(f"READER_FAILS_WITH_SUPPORT: {len(reader_fails)}/{total_with_evidence} ({100*len(reader_fails)/total_with_evidence:.1f}%)")
    print()

# Show examples of reader fails with evidence
print("Examples: evidence in input but Reader wrong:")
for p in reader_fails[:5]:
    print(f"  {p['question_id']} [{p['category']}]")
    print(f"    Expected: {p['expected']}")
    print(f"    Reader: {p['treatment_reader'][:70]}")
    print(f"    Final: {p['treatment_final'][:50]}")
    print()

# Save analysis
out = {
    "recovered": len(recovered),
    "regressed": len(regressed),
    "recovery_causes": cause_counts,
    "reader_ok_with_evidence": len(reader_ok),
    "reader_fails_with_evidence": len(reader_fails),
    "no_evidence_fails": len(no_evidence),
}
path = PROJECT_ROOT / "results/m30_38/analysis.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {path}")
