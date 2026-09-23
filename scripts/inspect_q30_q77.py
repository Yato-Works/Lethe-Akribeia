#!/usr/bin/env python3
"""Inspect the evidence for Q30 and Q77."""
import sys, json
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)

targets = ['conv-26-qa-030', 'conv-26-qa-077']
qmap = {q.question_id: q for q in questions if q.question_id in targets}

# Show evidence turns
for qid in targets:
    q = qmap[qid]
    print(f"\n=== {qid} ===")
    print(f"Q: {q.question}")
    print(f"GT: {q.ground_truth}")
    print(f"Evidence IDs: {q.evidence_ids}")
    
    # Find evidence turns
    for evid in q.evidence_ids:
        dia_id = evid.split(':')[0] if ':' in evid else evid
        turn_num = int(evid.split(':')[1]) if ':' in evid else 0
        for t in turns:
            if dia_id in t.dia_id:
                print(f"  Evidence turn {t.dia_id}: [{t.speaker}] {t.text[:120]}")
                break

# Also compile and check what context is provided
pcc30 = adapter.compiler.compile(qmap['conv-26-qa-030'].question, ir)
print(f"\n=== Q30 Context (first 500 chars) ===\n{pcc30.context_text[:500]}")

pcc77 = adapter.compiler.compile(qmap['conv-26-qa-077'].question, ir)
print(f"\n=== Q77 Context (first 500 chars) ===\n{pcc77.context_text[:500]}")
