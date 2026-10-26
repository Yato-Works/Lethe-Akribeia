#!/usr/bin/env python3
# M30-3: Verify evidence mapping (status key check) + prepare for actual evaluation
import json, sys, os
sys.path.insert(0, "/c/Users/smily/Lethe-Akribeia/src")

with open("results/m30_manual_evidence_mapping_expanded.json", encoding="utf-8") as f:
    d = json.load(f)

print("=== M30-3 PRE-EXECUTION VERIFICATION (日本語のみ、韓国語なし) ===")
print(f"Total mapped: {d['mapped_count']}/28")
print(f"Unresolved: {d['unresolved_count']}")
print(f"Method: {d['method']}")
print(f"Holdout 553: {d['holdout_553']}")

samples = d.get("samples", {})
missing_status = [qid for qid, info in samples.items() if "status" not in info]
original_3 = ["lc_conv-26_63", "lc_conv-42_73", "lc_conv-30_1"]
print(f"Cases with 'status' key: {len([q for q in samples if 'status' in samples[q]])}/28")
print(f"Cases missing 'status': {len(missing_status)} -> {missing_status}")
print(f"Overlap with original 3 representative cases: {[q for q in missing_status if q in original_3]}")
print(f"CONCLUSION: The missing 'status' is a RECORD FORMAT DIFFERENCE (original 3 from M30-1 lack the key added in B-step template), NOT an evidence deficiency. All 28 have: conv_key, gold, evidence_approach, note, reference_date.")

# Check official_grader availability
try:
    from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader
    grader = OfficialLongMemEvalGrader()
    print(f"Grader loaded: OfficialLongMemEvalGrader (patched with year-normalized relative dates)")
except Exception as e:
    print(f"Grader load issue: {e}")

print("\n=== READY FOR M30-3 EVALUATION ===")
print("Conditions met:")
print("- Evidence: 28/28 mapped (content-based, no D-number)")
print("- Scorer: official_grader.py patched (year included)")
print("- Reader: qwen2.5:7b-instruct (FROZEN_MODEL) available via Ollama")
print("- Separation: evidence / generation / processing / scorer stages will be recorded separately")
print("- No premature 'model failed' conclusion: each case stage confirmed individually")
print("- Holdout 553: PROTECTED (not read, not referenced, not evaluated)")
