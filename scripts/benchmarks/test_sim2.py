EXPANSION_MAP = {
    "children": ["son", "daughter", "kids", "kid", "baby", "babies", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "child": ["son", "daughter", "kids", "kid", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "kid": ["son", "daughter", "child", "children", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "kids": ["son", "daughter", "child", "children", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
}

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
adapter.compiler.rescue_order = "relevance"
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q = next(q for q in questions if q.question_id == 'conv-41-qa-040')
words = q.question.lower().replace('?', '').split()
extra_words = []
for w in words:
    if w in EXPANSION_MAP:
        extra_words.extend(EXPANSION_MAP[w])
expanded_q = q.question + " " + " ".join(extra_words)

pcc = adapter.compiler.compile(expanded_q, ir_records)
print("In compiled context (with relevance order):")
print("  Kyle:", "kyle" in pcc.context_text.lower())
print("  Sara:", "sara" in pcc.context_text.lower())
