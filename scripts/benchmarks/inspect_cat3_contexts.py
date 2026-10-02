import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()

# 1. conv-42-qa-004 (pets for Joanna)
turns42, q42, ir42 = adapter.load_conversation(conv_idx=3)
q004 = next(q for q in q42 if q.question_id == 'conv-42-qa-004')
pcc004 = adapter.compiler.compile(q004.question, ir42)
print("=== conv-42-qa-004 ===")
print("Q:", q004.question)
print("GT:", q004.ground_truth)
print("Has 'hairless' in pcc:", "hairless" in pcc004.context_text.lower())
print("Has 'pig' in pcc:", "pig" in pcc004.context_text.lower())
print("Evidence turns:", q004.evidence_ids)
for evid in q004.evidence_ids:
    t = next((t for t in turns42 if t.dia_id == evid), None)
    if t:
        print(f"  [Evidence {evid}] {t.speaker}: {t.text}")

# 2. conv-42-qa-085 (filmmaker vs movie producer)
q085 = next(q for q in q42 if q.question_id == 'conv-42-qa-085')
pcc085 = adapter.compiler.compile(q085.question, ir42)
print("\n=== conv-42-qa-085 ===")
print("Q:", q085.question)
print("GT:", q085.ground_truth)
print("Has 'filmmaker' in pcc:", "filmmaker" in pcc085.context_text.lower())
print("Evidence turns:", q085.evidence_ids)
for evid in q085.evidence_ids:
    t = next((t for t in turns42 if t.dia_id == evid), None)
    if t:
        print(f"  [Evidence {evid}] {t.speaker}: {t.text}")

# 3. conv-26-qa-059 (Would Caroline be considered religious?)
turns26, q26, ir26 = adapter.load_conversation(conv_idx=0)
q059 = next(q for q in q26 if q.question_id == 'conv-26-qa-059')
pcc059 = adapter.compiler.compile(q059.question, ir26)
print("\n=== conv-26-qa-059 ===")
print("Q:", q059.question)
print("GT:", q059.ground_truth)
print("Has 'religious' in pcc:", "religious" in pcc059.context_text.lower())
print("Evidence turns:", q059.evidence_ids)
for evid in q059.evidence_ids:
    t = next((t for t in turns26 if t.dia_id == evid), None)
    if t:
        print(f"  [Evidence {evid}] {t.speaker}: {t.text}")
