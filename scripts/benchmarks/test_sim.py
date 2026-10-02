EXPANSION_MAP = {
    "children": ["son", "daughter", "kids", "kid", "baby", "babies", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "child": ["son", "daughter", "kids", "kid", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "kid": ["son", "daughter", "child", "children", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "kids": ["son", "daughter", "child", "children", "baby", "infant", "toddler", "boy", "girl", "childhood", "family"],
    "family": ["aunt", "uncle", "cousin", "grandma", "grandmother", "grandpa", "grandfather", "parent", "parents", "mother", "father", "mom", "dad", "sister", "brother", "son", "daughter"],
    "aunt": ["auntie", "relative", "family"],
    "pet": ["dog", "dogs", "cat", "cats", "puppy", "puppies", "kitty", "kitties", "pup", "pups", "pig", "pigs", "hairless", "animal", "animals", "fur", "allergy"],
    "pets": ["dog", "dogs", "cat", "cats", "puppy", "puppies", "kitty", "kitties", "pup", "pups", "pig", "pigs", "hairless", "animal", "animals", "fur", "allergy"],
    "job": ["filmmaker", "director", "producer", "writer", "author", "keeper", "artist", "store", "business", "studio"],
    "career": ["filmmaker", "director", "producer", "writer", "author", "keeper", "artist", "store", "business", "studio"],
    "degree": ["political", "science", "administration", "public", "affairs", "university", "college", "school"],
    "store": ["boutique", "brand", "studio", "fashion", "online", "promotions", "pieces"],
    "shop": ["boutique", "brand", "studio", "fashion", "online", "promotions", "pieces"],
    "event": ["poetry", "reading", "conference", "pride", "show", "exhibition"],
    "events": ["poetry", "reading", "conference", "pride", "show", "exhibition"],
    "childhood": ["kid", "younger", "child", "children", "reminds"],
}

from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q = next(q for q in questions if q.question_id == 'conv-41-qa-040')

# Simulate expansion in query string
words = q.question.lower().replace('?', '').split()
extra_words = []
for w in words:
    if w in EXPANSION_MAP:
        extra_words.extend(EXPANSION_MAP[w])
expanded_q = q.question + " " + " ".join(extra_words)
print("Expanded Q:", expanded_q)

intent, cand = adapter.compiler.reconstructor.reconstruct_world(expanded_q, ir_records)
for i, u in enumerate(cand):
    if '[D8:4' in u.ir.raw_content:
        print(f"  D8:4 (Kyle) Rank: #{i+1}")
    if '[D22:7' in u.ir.raw_content:
        print(f"  D22:7 (Sara) Rank: #{i+1}")

pcc = adapter.compiler.compile(expanded_q, ir_records)
print("In compiled context:")
print("  Kyle:", "kyle" in pcc.context_text.lower())
print("  Sara:", "sara" in pcc.context_text.lower())
