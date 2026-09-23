#!/usr/bin/env python3
"""Debug the _parse_turns method."""
import sys, json
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)

q = next(q for q in questions if q.question_id == 'conv-26-qa-077')
pcc = adapter.compiler.compile(q.question, ir)

parsed = adapter.compiler.persona_store._parse_turns(pcc.context_text)
print(f"Parsed turns: {len(parsed)}")
for i, (spk, text) in enumerate(parsed):
    print(f"  {i}: Speaker='{spk.strip()[:20]}', Text='{text.strip()[:80]}'")
    if i > 10:
        break
