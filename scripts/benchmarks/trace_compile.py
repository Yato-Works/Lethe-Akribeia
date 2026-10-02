from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph

adapter = LoCoMoAdapter()
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q = next(q for q in questions if q.question_id == 'conv-41-qa-018')
print("Question:", q.question)

intent, candidate_units = adapter.compiler.reconstructor.reconstruct_world(q.question, ir_records)
graph = UnifiedPropositionGraph()
graph.build_from_records(ir_records)

widened = adapter.compiler._apply_evidence_widening(q.question, candidate_units, ir_records, graph)
print("Candidate units count:", len(candidate_units))
print("Widened count:", len(widened))
print("Last rescue count:", adapter.compiler._last_rescue_count)

for i, u in enumerate(candidate_units[:25]):
    if 'rob' in u.ir.raw_content.lower() or 'D7:16' in u.ir.raw_content:
        print(f"  candidate_units #{i+1}: {u.ir.raw_content[:90]}")

for i, u in enumerate(widened[:25]):
    if 'rob' in u.ir.raw_content.lower() or 'D7:16' in u.ir.raw_content:
        print(f"  widened #{i+1}: {u.ir.raw_content[:90]}")

# Now let's see why it's not selected in compile loop:
pcc = adapter.compiler.compile(q.question, ir_records)
print("\nIs D7:16 in final context?", "D7:16" in pcc.context_text)
print("Is Rob in final context?", "rob" in pcc.context_text.lower())
