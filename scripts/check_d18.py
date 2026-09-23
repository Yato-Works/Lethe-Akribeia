#!/usr/bin/env python3
"""Check D18 turns for Q77."""
import json, sys, os
sys.path.insert(0, 'src')
os.chdir('/home/eli/Projects/artificial_memory')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)

# Check raw content of D18 turns in IR records
for r in ir:
    if 'D18' in r.raw_content[:20]:
        print(f"{r.raw_content[:250]}")
        print("---")
