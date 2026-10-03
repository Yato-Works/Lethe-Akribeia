#!/usr/bin/env python3
"""Run all 199 questions for Conv 0 with improved code."""
import sys
import json
import time
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)
answerer = OllamaAnswerer()

results_dir = 'benchmark_results/improved_v2_full'
import os
os.makedirs(results_dir, exist_ok=True)

results = []
for i, q in enumerate(questions):
    t0 = time.time()
    result = adapter.evaluate_question(q, turns, ir, answerer)
    elapsed = time.time() - t0

    correct = result.is_correct
    pred = result.predicted_answer
    oracle = result.oracle_recall

    if i % 10 == 0:
        print(f"[{i}/{len(questions)}] Q{q.question_id[-4:]} (cat {q.category}): correct={correct} oracle={oracle} ({elapsed:.0f}s)", flush=True)

    results.append({
        'question_id': q.question_id,
        'question': q.question,
        'category': q.category,
        'category_name': adapter.CATEGORY_NAMES.get(q.category, 'unknown'),
        'prediction': pred,
        'ground_truth': q.ground_truth,
        'correct': correct,
        'oracle_recall': oracle,
        'tokens_used': result.tokens_used,
        'latency_ms': result.latency_ms,
    })

    # Save progress every 10
    if i % 10 == 9:
        with open(f'{results_dir}/conv0_progress.jsonl', 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

# Final save
with open(f'{results_dir}/conv0_results.json', 'w') as f:
    json.dump({'conv_id': 'conv-26', 'total': len(questions), 'results': results}, f, indent=2)

# Summary by category
from collections import Counter
cat_correct = Counter()
cat_total = Counter()
for r in results:
    cat_total[r['category']] += 1
    if r['correct']:
        cat_correct[r['category']] += 1

overall_correct = sum(1 for r in results if r['correct'])
print("\n=== CONV 0 RESULTS ===", flush=True)
print(f"Overall: {overall_correct}/{len(questions)} = {overall_correct/len(questions)*100:.1f}%", flush=True)
for cat in sorted(cat_total.keys()):
    name = adapter.CATEGORY_NAMES.get(cat, str(cat))
    c = cat_correct[cat]
    t = cat_total[cat]
    print(f"  {name} (cat {cat}): {c}/{t} = {c/t*100:.1f}%", flush=True)

oracle_total = sum(1 for r in results if r['oracle_recall'])
print(f"Oracle Recall: {oracle_total}/{len(questions)} = {oracle_total/len(questions)*100:.1f}%", flush=True)
