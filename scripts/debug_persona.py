#!/usr/bin/env python3
"""Debug persona summary for failed open-domain questions."""
import sys, json
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)

# Q30: "Would Melanie be considered a member of the LGBTQ community?"
# Q77: "Would Melanie go on another roadtrip soon?"
failed_qids = ['conv-26-qa-030', 'conv-26-qa-077']
pccs = {}
for q in questions:
    if q.question_id in failed_qids:
        pcc = adapter.compiler.compile(q.question, ir)
        summary = adapter.compiler.persona_store.get_persona_summary(q.question, pcc.context_text)
        print(f"\n=== Q{q.question_id[-4:]} ===")
        print(f"Q: {q.question}")
        print(f"GT: {q.ground_truth}")
        print(f"Evidence IDs: {q.evidence_ids}")
        print(f"\nPersona Summary: {summary}")
        print(f"\nContext (first 1000 chars): {pcc.context_text[:1000]}")
        print(f"Context contains 'ally'?: {'ally' in pcc.context_text.lower()}")
        print(f"Context contains 'identify'?: {'identify' in pcc.context_text.lower()}")
        print(f"Context contains 'member'?: {'member' in pcc.context_text.lower()}")
