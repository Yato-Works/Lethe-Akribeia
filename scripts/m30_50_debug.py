#!/usr/bin/env python3
"""Debug: test the post_process on lc_conv-42_57's reader output."""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
exec(open("scripts/m30_50_phase1_postprocess.py").read().split("def main")[0])

prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
bp = next(r for r in prod["rows"] if r["qid"] == "lc_conv-42_57")
raw = bp["predicted"]
print("Reader output:")
print(repr(raw))
print()

# Check pattern matches
for pat, label in rel_patterns:
    m = re.search(pat, raw, re.I)
    if m:
        print(f"Pattern '{label}': matched, groups={m.groups()}")
    else:
        print(f"Pattern '{label}': no match")

# Now with context
locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
turns, questions, ir_records = adapter.load_conversation(sids.index("conv-42"))
q = next(q for q in questions if q.question_id == "conv-42-qa-057")
pcc = adapter.compiler.compile(q.question, ir_records)
ctx = pcc.context_text

print("\nContext lines with 'next day' or 'tomorrow':")
for line in ctx.split("\n"):
    if "next day" in line.lower() or "tomorrow" in line.lower():
        header = line[:80]
        inctx = post_process_temporal(raw, ctx, q.question)
        print(f"  {header}")

result = post_process_temporal(raw, ctx, q.question)
print(f"\nPost-process result: {result!r}")

# Also check the evidence turn
for line in ctx.split("\n"):
    if "D26:19" in line:
        print(f"\nEvidence turn: {line[:200]}")
