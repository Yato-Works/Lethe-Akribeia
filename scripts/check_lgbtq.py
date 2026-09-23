#!/usr/bin/env python3
"""Find Melanie's LGBTQ identity statements in Conv 0."""
import sys, json, re
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)

# Search for LGBTQ-related content in all turns
print("=== All LGBTQ-related turns ===")
for i, t in enumerate(turns):
    tl = t.text.lower()
    if any(w in tl for w in ['lgbtq', 'lgbt', 'transgender', 'gay', 'lesbian', 'ally', 'identify as', 'part of']):
        print(f"  Dia {t.dia_id} [{t.speaker}]: {t.text[:120]}")

# Check what MSC returns for Q30
q30 = next(q for q in questions if q.question_id == 'conv-26-qa-030')
pcc = adapter.compiler.compile(q30.question, ir)
print(f"\n=== Q30 MSC Context ===\n{pcc.context_text[:1000]}")
print(f"\nToken cost: {pcc.token_cost}")
