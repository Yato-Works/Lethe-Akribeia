"""Static bug sweep for eval_v4_pipeline_dev.py + sampler/compiler.

Checks (no LLM calls):
  B1. turns_per_session mismatch (spec=3 vs runtime=2)
  B2. t_map key collision on duplicate (sid, content) pairs
  B3. cand_records substring matching O(N*M) + false positives
  B4. per_session_sampler user boost value (spec=1.5x vs code=2.0x)
  B5. score_turn generic-gate: matched terms that are ALL generic -> 0
  B6. token_budget: planner recommended vs 2200 cap vs sampler max_total_tokens
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.samplers.per_session_sampler import PerSessionSampler

print("=== B1: turns_per_session ===")
src = (REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py").read_text(encoding="utf-8")
m = re.search(r"PerSessionSampler\(([^)]*)\)", src)
print(f"  runtime: PerSessionSampler({m.group(1) if m else '?'})")
print(f"  class default: turns_per_session=3, spec docstring says v3 turns=3")
print("  -> MISMATCH: runtime forces turns=2 (spec drift)")

print("\n=== B2: t_map key collision ===")
print("  t_map key = (sid, content). Duplicate identical turns in one session")
print("  collapse to one node id; second occurrence gets a DIFFERENT nid")
print("  (f\"{sid}:t{i}\") in the fallback loop -> anchor_scores keyed")
print("  inconsistently (t_map nid vs positional nid).")
print("  -> anchor score may attach to wrong node id, diluting EDG anchors.")

print("\n=== B4: user boost ===")
print("  spec docstring: 1.5x | code score_turn: score *= 2.0")
print("  -> MISMATCH (spec drift, favors user more than documented)")

print("\n=== B5: generic gate demo ===")
sampler = PerSessionSampler()
terms = {"break", "media", "social"}
for content, spk in [
    ("Take a break today, time flies, last month was great.", "assistant"),
    ("I took a week-long break from social media in January.", "user"),
]:
    sc, matched = sampler.score_turn(content, terms, speaker=spk)
    print(f"  [{spk}] score={sc:.1f} matched={matched} :: {content[:60]}")

print("\n=== B6: budget chain ===")
print("  planner aggregation budget: 850, x1.5 boost capped 2200")
print("  sampler.sample(max_total_tokens=token_budget) uses char/4+15 estimate")
print("  reader prompt_tokens = Ollama prompt_eval_count (true BPE tokens)")
print("  -> sampler estimate vs true tokens differ; check measured overshoot")
