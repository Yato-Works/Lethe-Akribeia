#!/usr/bin/env python3
"""Debug: check the actual context format."""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
turns, questions, ir_records = adapter.load_conversation(sids.index("conv-42"))
q = next(q for q in questions if q.question_id == "conv-42-qa-057")
pcc = adapter.compiler.compile(q.question, ir_records)
ctx = pcc.context_text

# Print first 20 lines to see format
lines = ctx.split("\n")
print(f"Total context lines: {len(lines)}")
print("First 25 lines:")
for i, line in enumerate(lines[:25]):
    print(f"  {i}: {line[:120]}")

print("\nSearching for 'D26:19':")
for i, line in enumerate(lines):
    if "D26:19" in line:
        print(f"  {i}: {line[:150]}")

print("\nSearching for 'tomorrow':")
for i, line in enumerate(lines):
    if "tomorrow" in line.lower() or "next day" in line.lower():
        print(f"  {i}: {line[:120]}")
