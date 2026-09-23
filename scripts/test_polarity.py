#!/usr/bin/env python3
"""Test just the 2 failed open-domain questions."""
import sys, json, time
sys.path.insert(0, 'src')
from artificial_memory.research.benchmarks.external.locomo_adapter import LoCoMoAdapter
from artificial_memory.research.benchmarks.llm import OllamaAnswerer, FROZEN_MODEL
from artificial_memory.recall.evidence_scorer import EvidenceScoreWeights

adapter = LoCoMoAdapter()
answerer = OllamaAnswerer(model=FROZEN_MODEL, timeout_seconds=600.0)
weights = EvidenceScoreWeights()
turns, questions, ir = adapter.load_conversation(0)

targets = ['conv-26-qa-030', 'conv-26-qa-077']
for q in questions:
    if q.question_id in targets:
        print(f'Q: {q.question}', flush=True)
        print(f'Evidence: {q.evidence_ids}', flush=True)
        print(f'GT: {q.ground_truth}', flush=True)
        t0 = time.time()
        res = adapter.evaluate_question(q, turns, ir, answerer, weights=weights)
        elapsed = time.time() - t0
        print(f'Pred: {res.predicted_answer[:80]}', flush=True)
        print(f'Correct: {res.is_correct} | Oracle: {res.oracle_recall} | {elapsed:.1f}s', flush=True)
        print()
