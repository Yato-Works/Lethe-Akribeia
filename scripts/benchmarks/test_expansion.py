from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter

adapter = LoCoMoAdapter()
conv_idx = 2
turns, questions, ir_records = adapter.load_conversation(conv_idx=conv_idx)

q_expanded = "What are the names of John's children son daughter kids kid baby toddler one-year-old boy girl named?"
intent, cand = adapter.compiler.reconstructor.reconstruct_world(q_expanded, ir_records)
print("--- Top 15 candidates ---")
for i in range(15):
    print(f"#{i+1}: {cand[i].ir.raw_content[:90]}")
