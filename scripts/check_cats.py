#!/usr/bin/env python3
import sys, json
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)

cats = {}
for q in questions:
    cats[q.category] = cats.get(q.category, 0) + 1
print("Categories:", cats)
print(f"Total questions: {len(questions)}")
print(f"Question IDs: {[q.question_id for q in questions[:5]]}")
