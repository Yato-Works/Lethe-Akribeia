from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q = next(q for q in questions if q.question_id == 'conv-41-qa-018')
print("Question:", q.question)

intent, candidate_units = adapter.compiler.reconstructor.reconstruct_world(q.question, ir_records)
for i, u in enumerate(candidate_units):
    if 'rob' in u.ir.raw_content.lower() and 'yoga' in u.ir.raw_content.lower():
        print(f"  Candidate #{i+1}: {u.ir.raw_content[:100]}")
