from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights

adapter = LoCoMoAdapter()
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q = next(q for q in questions if q.question_id == 'conv-41-qa-040')
print("Question:", q.question)

intent, candidate_units = adapter.compiler.reconstructor.reconstruct_world(q.question, ir_records)

# Find D8:4 in candidate_units
for i, u in enumerate(candidate_units):
    if '[D8:4' in u.ir.raw_content:
        print(f"D8:4 Rank: #{i+1}")
        print(f"Content: {u.ir.raw_content}")
        print(f"IR Source: {u.ir.source}, entity: {u.entity}, prop: {u.target_property}, val: {u.ir.value}")
    if '[D22:7' in u.ir.raw_content or '[D22:8' in u.ir.raw_content:
        print(f"D22 Rank: #{i+1} : {u.ir.raw_content[:80]}")
