#!/usr/bin/env python3
"""Run all open-domain (cat 3) questions for Conv 0 with improved code."""
import sys
import json
import time
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

adapter = LoCoMoAdapter()
turns, questions, ir = adapter.load_conversation(0)
answerer = OllamaAnswerer()

open_domain = [q for q in questions if q.category == 3]
print(f"Open-domain questions (cat 3): {len(open_domain)}", flush=True)

results = []
for q in open_domain:
    t0 = time.time()
    result = adapter.evaluate_question(q, turns, ir, answerer)
    elapsed = time.time() - t0

    correct = result.is_correct
    pred = result.predicted_answer
    oracle = result.oracle_recall
    print(f"Q{q.question_id[-4:]}: pred='{pred[:60]}' gt='{q.ground_truth[:50]}' oracle={oracle} correct={correct} ({elapsed:.0f}s)", flush=True)

    results.append({
        'question_id': q.question_id,
        'question': q.question,
        'category': q.category,
        'prediction': pred,
        'ground_truth': q.ground_truth,
        'correct': correct,
        'oracle_recall': oracle,
        'tokens_used': result.tokens_used,
        'elapsed': round(elapsed, 1),
    })

with open('benchmark_results/open_domain_v4.jsonl', 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

correct_count = sum(1 for r in results if r['correct'])
total = len(open_domain)
print(f"\n=== RESULTS: {correct_count}/{total} = {correct_count/total*100:.1f}% ===", flush=True)
