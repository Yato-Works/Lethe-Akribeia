#!/usr/bin/env python3
"""Check if Q77 MSC context has the accident evidence."""
import sys, json
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)

q77 = next(q for q in questions if q.question_id == 'conv-26-qa-077')
pcc = adapter.compiler.compile(q77.question, ir)

print(f"Evidence IDs: {q77.evidence_ids}")
print(f"Token cost: {pcc.token_cost}")
for ev in q77.evidence_ids:
    in_ctx = ev in pcc.context_text
    print(f"  {ev} in context: {in_ctx}")

print(f"\nFull context:\n{pcc.context_text}")
