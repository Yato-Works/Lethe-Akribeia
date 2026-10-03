#!/usr/bin/env python3
"""Run retrieval-only for all 8 dev convs, then LLM only on oracle=True questions."""
import sys
import json
import time
import os
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer

adapter = LoCoMoAdapter()
answerer = OllamaAnswerer()

dev_convs = [0, 1, 2, 4, 5, 6, 8, 9]
results_dir = 'benchmark_results/improved_v3'
os.makedirs(results_dir, exist_ok=True)

all_results = {}

for conv_idx in dev_convs:
    turns, questions, ir = adapter.load_conversation(conv_idx)
    print(f"\n=== Conv {conv_idx}: {len(questions)} questions ===", flush=True)

    # Retrieval-only first
    conv_results = []
    for q in questions:
        pcc = adapter.compiler.compile(q.question, ir)
        oracle = False
        if q.evidence_ids:
            oracle = any(ev_id in pcc.context_text for ev_id in q.evidence_ids)
        else:
            oracle = True
        conv_results.append({
            'question_id': q.question_id,
            'question': q.question,
            'category': q.category,
            'ground_truth': q.ground_truth,
            'evidence_ids': q.evidence_ids,
            'oracle_recall': oracle,
            'token_cost': pcc.token_cost,
            'context_text': pcc.context_text,
            'is_abstention': pcc.is_abstention,
        })

    # Count by category
    from collections import Counter
    cat_oracle = Counter()
    cat_total = Counter()
    for r in conv_results:
        cat_total[r['category']] += 1
        if r['oracle_recall']:
            cat_oracle[r['category']] += 1
    
    total_oracle = sum(1 for r in conv_results if r['oracle_recall'])
    print(f"  Oracle recall: {total_oracle}/{len(conv_results)} ({total_oracle/len(conv_results)*100:.1f}%)", flush=True)
    for cat in sorted(cat_total.keys()):
        name = adapter.CATEGORY_NAMES.get(cat, str(cat))
        print(f"    {name} (cat {cat}): oracle={cat_oracle[cat]}/{cat_total[cat]} ({cat_oracle[cat]/cat_total[cat]*100:.1f}%)", flush=True)

    # Save retrieval results
    with open(f'{results_dir}/conv{conv_idx}_retrieval.json', 'w') as f:
        json.dump({'conv_idx': conv_idx, 'total': len(conv_results), 'results': conv_results}, f, indent=2)
    
    all_results[conv_idx] = conv_results

print("\n=== RETRIEVAL-ONLY COMPLETE ===", flush=True)
for conv_idx in dev_convs:
    r = all_results[conv_idx]
    total = len(r)
    oracle = sum(1 for x in r if x['oracle_recall'])
    print(f"Conv {conv_idx}: oracle {oracle}/{total} ({oracle/total*100:.1f}%)", flush=True)
