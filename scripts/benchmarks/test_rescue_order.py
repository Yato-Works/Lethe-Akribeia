from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q = next(q for q in questions if q.question_id == 'conv-41-qa-018')

# Test default
pcc1 = adapter.compiler.compile(q.question, ir_records)
print("Default 'pool': D7:16 in context?", "D7:16" in pcc1.context_text, "Rob in context?", "rob" in pcc1.context_text.lower())

# Test relevance rescue order
adapter.compiler.rescue_order = "relevance"
pcc2 = adapter.compiler.compile(q.question, ir_records)
print("Relevance order: D7:16 in context?", "D7:16" in pcc2.context_text, "Rob in context?", "rob" in pcc2.context_text.lower())

# Test with strong = 12
# Let's see what happens to all questions in conv-41
print("\n--- conv-41 test ---")
for qid in ['conv-41-qa-007', 'conv-41-qa-009', 'conv-41-qa-018', 'conv-41-qa-024', 'conv-41-qa-040']:
    q_obj = next(q for q in questions if q.question_id == qid)
    p = adapter.compiler.compile(q_obj.question, ir_records)
    print(f"[{qid}] len={len(p.context_text)}, 'rob'={('rob' in p.context_text.lower())}, 'doll'={('doll' in p.context_text.lower())}, 'kyle'={('kyle' in p.context_text.lower())}")
