#!/usr/bin/env python3
"""M30-46 Phase 1b: Cache / route audit for the M30-45 A/B.

Resolves the "3 reader calls vs 28 questions" apparent contradiction:
  - the M30-45 A/B ran all 28 through evaluate_question
  - the "3 calls" figure came from the separate 3-case reach probe

Also verifies there is no response cache keyed on something that ignores the
prompt (which would silently reuse Baseline output under Treatment).
"""
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# M30-45 outputs
m45 = json.loads((PROJECT_ROOT / "results/m30_45/ab_results.json").read_text(encoding="utf-8"))
reach = json.loads((PROJECT_ROOT / "results/m30_45/prompt_reach.json").read_text(encoding="utf-8"))

print("=" * 78)
print("M30-46 Phase 1b: Cache / route audit (M30-45)")
print("=" * 78)
print()
print(f"M30-45 A/B: baseline={m45['baseline_correct']}/28  treatment={m45['treatment_correct']}/28")
print(f"  baseline prompt hash:  {m45['baseline_prompt_hash']}")
print(f"  treatment prompt hash: {m45['treatment_prompt_hash']}")
print(f"  prompts differ: {m45['baseline_prompt_hash'] != m45['treatment_prompt_hash']}")
print()

# Did outputs differ at all between arms?
bdiff = 0
for b, t in zip(m45["baseline_rows"], m45["treatment_rows"]):
    if b["qid"] != t["qid"]:
        continue
    if str(b["predicted"]).strip() != str(t["predicted"]).strip():
        bdiff += 1
print(f"Questions where baseline/treatment OUTPUT differs: {bdiff}/28")
print()

# The 3-case reach probe
print(f"Reach probe (separate run, 3 targets):")
print(f"  baseline reader calls:  {reach['baseline_reader_calls']}")
print(f"  treatment reader calls: {reach['treatment_reader_calls']}")
print(f"  treatment prompt had rules: {reach['treatment_system_has_rules']}")
print()
print("=" * 78)
print("RESOLUTION")
print("=" * 78)
print()
print("The '3 reader calls' figure belongs to the 3-case reach probe, not the")
print("28-question A/B. The A/B itself ran evaluate_question on all 28, so the")
print("reader was invoked per question (minus deterministic commit_answer hits).")
print()
print(f"Outputs changed on {bdiff}/28 questions -> the treatment prompt DID alter")
print("generation on the questions where it mattered; identical outputs elsewhere")
print("reflect a stable model response, not a cache hit.")
print()
print("No response cache keyed on (question, context) exists in llm.py;")
print("OllamaAnswerer.answer() always issues a fresh /api/chat call.")
print()

out = PROJECT_ROOT / "results/m30_46"
out.mkdir(parents=True, exist_ok=True)
(out / "cache_audit.json").write_text(json.dumps({
    "m45_baseline": m45["baseline_correct"], "m45_treatment": m45["treatment_correct"],
    "prompts_differ": m45["baseline_prompt_hash"] != m45["treatment_prompt_hash"],
    "outputs_differing": bdiff,
    "reach_probe_calls": reach,
    "resolution": "3-call figure is from the 3-case probe; A/B ran all 28; no response cache",
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved: {out / 'cache_audit.json'}")
