from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights

adapter = LoCoMoAdapter()
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q = next(q for q in questions if q.question_id == 'conv-41-qa-040')
print("Question:", q.question)

# Inspect all records from D8
d8_recs = [r for r in ir_records if '[D8:' in r.raw_content]
print(f"Total D8 records: {len(d8_recs)}")

for r in d8_recs:
    print(f"\n{r.raw_content[:150]}")
    # Compute score
    # Look at unit_relevance logic
