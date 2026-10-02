EXPANSION_MAP = {
    "children": ["son", "daughter", "kids", "kid", "baby", "babies", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "child": ["son", "daughter", "kids", "kid", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "kid": ["son", "daughter", "child", "children", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "kids": ["son", "daughter", "child", "children", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
}

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.recall.proposition_graph import UnifiedPropositionGraph

adapter = LoCoMoAdapter()
adapter.compiler.selection_window_cap = 36
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q = next(q for q in questions if q.question_id == 'conv-41-qa-040')
words = q.question.lower().replace('?', '').split()
extra_words = []
for w in words:
    if w in EXPANSION_MAP:
        extra_words.extend(EXPANSION_MAP[w])
expanded_q = q.question + " " + " ".join(extra_words)

intent, candidate_units = adapter.compiler.reconstructor.reconstruct_world(expanded_q, ir_records)
graph = UnifiedPropositionGraph()
graph.build_from_records(ir_records)

# Test with relevance rescue order
adapter.compiler.rescue_order = "relevance"
widened = adapter.compiler._apply_evidence_widening(expanded_q, candidate_units, ir_records, graph)
print(f"Total candidate units: {len(candidate_units)}, widened: {len(widened)}")
for i, u in enumerate(widened[:36]):
    if 'Sara' in u.ir.raw_content:
        print(f"  Sara in widened at #{i+1}: {u.ir.raw_content[:80]}")
    if 'Kyle' in u.ir.raw_content:
        print(f"  Kyle in widened at #{i+1}: {u.ir.raw_content[:80]}")

pcc = adapter.compiler.compile(expanded_q, ir_records)
print("In compiled context:")
print("  Kyle:", "kyle" in pcc.context_text.lower())
print("  Sara:", "sara" in pcc.context_text.lower())
