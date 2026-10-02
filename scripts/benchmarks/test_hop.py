EXPANSION_MAP = {
    "children": ["son", "daughter", "kids", "kid", "baby", "babies", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "child": ["son", "daughter", "kids", "kid", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "kid": ["son", "daughter", "child", "children", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "kids": ["son", "daughter", "child", "children", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
}

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)
q = next(q for q in questions if q.question_id == 'conv-41-qa-040')
words = q.question.lower().replace('?', '').split()
extra_words = []
for w in words:
    if w in EXPANSION_MAP:
        extra_words.extend(EXPANSION_MAP[w])
expanded_q = q.question + ' ' + ' '.join(extra_words)

rec = adapter.compiler.reconstructor
units = rec.tag_memory_roles(ir_records)

scored = []
for u in units:
    b = rec.evidence_scorer.compute_score(expanded_q, u, None)
    sc = b.total_score
    scored.append((sc, u))

scored.sort(key=lambda x: x[0], reverse=True)
print("--- Raw unit_relevance scores (before hop_boost) ---")
for i in range(10):
    sc, u = scored[i]
    print(f"#{i+1} [score {sc:.1f}]: {u.ir.raw_content[:85]}")

for i, (sc, u) in enumerate(scored):
    if 'Kyle' in u.ir.raw_content:
        print(f"Kyle at rank #{i+1} with score {sc:.1f}")
    if 'Sara' in u.ir.raw_content:
        print(f"Sara at rank #{i+1} with score {sc:.1f}")
