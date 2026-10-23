#!/usr/bin/env python3
"""M30-49 Phase 4: Verify post-processing CAN fix the 1 actionable case.

lc_conv-42_57: reader said "Joanna planned to go over to Nate's place the next day
(10 November 2022) to sha..." and gold is "5 November, 2022."

The reader actually got the right turn date (Nov 10) but wrong direction
(the next day should be Nov 11, but gold = Nov 5).

Actually let's check: does the reader output contain 10 November 2022 as a
fact from the context? If so, the issue is different than assumed.
"""
import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

prod = json.loads((PROJECT_ROOT / "results/m30_43/production_bucketA.json").read_text(encoding="utf-8"))
bp = next(r for r in prod["rows"] if r["qid"] == "lc_conv-42_57")

print(f"QID: lc_conv-42_57")
print(f"Gold: {bp['gold']!r}")
print(f"Reader output: {bp['predicted']!r}")
print(f"Correct: {bp['correct']}")
print()

# What's in the context?
locomo = json.loads((PROJECT_ROOT / "datasets/external/locomo10.json").read_text(encoding="utf-8"))
sids = [c.get("sample_id", "") for c in locomo]
adapter = LoCoMoAdapter(dataset_path=str(PROJECT_ROOT / "datasets/external/locomo10.json"))
turns, questions, ir_records = adapter.load_conversation(sids.index("conv-42"))

q = next(q for q in questions if q.question_id == "conv-42-qa-057")
pcc = adapter.compiler.compile(q.question, ir_records)
ctx = pcc.context_text

# Find D26:19
for line in ctx.split("\n"):
    if "D26:19" in line:
        print(f"Evidence turn in context: {line[:200]}")
        break

# Find the header date
m = re.search(r"D26:19.*?(on\s+\w+,\s+\d{4})", ctx)
if m:
    print(f"Header date: {m.group(1)}")
    
# The question
print(f"\nQuestion: {q.question}")
print(f"\nReader mentions '10 November 2022' — but the turn header should be 4 November 2022")
print(f"Tomorrow from Nov 4 = Nov 5. Gold = 5 November 2022.")
print(f"So the reader has the right RELATIONSHIP (tomorrow) but wrong anchor,")
print(f"OR the context has both Nov 4 and Nov 10 dates.")
