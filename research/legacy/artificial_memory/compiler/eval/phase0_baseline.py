#!/usr/bin/env python3
"""
Phase 0 Runner: Label-Free Baseline Evaluation for Lethe-Akribeia V4.

This script:
1. Loads Dev split (250 questions)
2. Runs label-free type classifier (no gold labels)
3. Runs full V4.2 pipeline with predicted types
4. Grades with official LongMemEval grader
5. Computes evidence recall @ budget (no LLM calls)
6. Outputs statistical report with confidence intervals

Usage:
    python -m artificial_memory.compiler.eval.phase0_baseline \
        --dev-path benchmark_results/splits/lme_dev_qids.json \
        --haystack-path benchmark_results/splits/lme_dev_full.json \
        --output benchmark_results/phase0_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from artificial_memory.compiler.eval.evidence_recall import EvidenceRecallEvaluator, load_dev_split
from artificial_memory.compiler.eval.official_grader import OfficialLongMemEvalGrader
from artificial_memory.compiler.eval.stat_test import StatisticalTester
from artificial_memory.compiler.pipeline import CognitiveCompilerPipeline
from artificial_memory.compiler.router.label_free_classifier import LabelFreeTypeClassifier, load_dev_items


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Label-Free Baseline Evaluation")
    parser.add_argument("--dev-path", type=str, required=True, help="Path to Dev split (qids)")
    parser.add_argument("--haystack-path", type=str, required=True, help="Path to full Dev haystack data")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--budget", type=int, default=1300, help="Token budget for V4.2 pipeline")
    parser.add_argument("--use-embeddings", action="store_true", help="Use embeddings for type classification")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Bootstrap iterations for CI")
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 0: LABEL-FREE BASELINE EVALUATION")
    print("=" * 70)
    print(f"Dev split: {args.dev_path}")
    print(f"Haystack: {args.haystack_path}")
    print(f"Budget: {args.budget} tokens")
    print(f"Use embeddings: {args.use_embeddings}")
    print()
    
    # 1. Load data
    print("Loading Dev data...")
    dev_qids = load_dev_items(args.dev_path)
    dev_full = load_dev_split(args.haystack_path)
    
    # Merge qids with full data
    full_by_id = {item['question_id']: item for item in dev_full}
    dev_items = []
    for qid_item in dev_qids:
        qid = qid_item['question_id']
        if qid in full_by_id:
            merged = {**full_by_id[qid], **qid_item}
            dev_items.append(merged)
        else:
            print(f"Warning: qid {qid} not found in haystack")
    
    print(f"Loaded {len(dev_items)} Dev questions")
    
    # 2. Initialize label-free classifier
    print("\nInitializing label-free type classifier...")
    classifier = LabelFreeTypeClassifier(use_embeddings=args.use_embeddings)
    
    # Evaluate classifier on Dev
    print("Evaluating type classifier...")
    cls_results = classifier.evaluate_on_dev(dev_items)
    print(f"  Type classification accuracy: {cls_results['accuracy']:.1%}")
    for qtype, metrics in cls_results['per_type'].items():
        if metrics['support'] > 0:
            print(f"    {qtype:30s} F1: {metrics['f1']:.3f}  support: {metrics['support']}")
    
    # 3. Initialize V4.2 pipeline
    print("\nInitializing V4.2 Cognitive Compiler Pipeline...")
    pipeline = CognitiveCompilerPipeline(
        token_budget=args.budget,
        use_deterministic_routing=True,  # Use predicted types
    )
    
    # 4. Run full pipeline with predicted types
    print(f"\nRunning V4.2 pipeline on {len(dev_items)} questions...")
    print(f"Token budget: {args.budget}")
    
    predictions = []
    start_time = time.time()
    
    for i, item in enumerate(dev_items):
        # Get predicted type
        pred_type = classifier.classify(item['question']).predicted_type
        
        # Run pipeline (this calls the LLM)
        result = pipeline.run(
            question=item['question'],
            sessions=item['haystack_sessions'],
            session_dates=item['haystack_dates'],
            session_ids=item['haystack_session_ids'],
            question_type=pred_type,  # Use PREDICTED type, not gold
            reference_date=item.get('question_date'),
        )
        
        predictions.append({
            'question_id': item['question_id'],
            'question_type': item['question_type'],  # Gold type for grading
            'predicted_type': pred_type,
            'question': item['question'],
            'predicted_answer': result.answer,
            'ground_truth': item.get('answer', ''),
            'prompt_tokens': result.prompt_tokens,
            'latency_ms': result.latency_ms,
            'is_correct': False,  # Will be filled by grader
        })
        
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            print(f"  {i+1}/{len(dev_items)} ({elapsed:.1f}s)")
    
    elapsed = time.time() - start_time
    print(f"Pipeline completed in {elapsed:.1f}s ({elapsed/len(dev_items):.2f}s/q)")
    
    # 5. Grade with official grader
    print("\nGrading with official LongMemEval grader...")
    grader = OfficialLongMemEvalGrader()
    graded = grader.grade_batch(predictions)
    
    # Update predictions with grading results
    for pred, grade in zip(predictions, graded):
        pred['is_correct'] = grade.is_correct
        pred['grading_details'] = grade.details
    
    # 6. Compute metrics
    metrics = grader.compute_metrics(graded)
    print(f"\nOverall Accuracy: {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})")
    for qtype, m in metrics['per_type'].items():
        print(f"  {qtype:30s} {m['accuracy']:.1%} ({m['correct']}/{m['total']})")
    
    # 7. Evidence Recall @ Budget (fast, no LLM)
    print("\nMeasuring Evidence Recall @ Budget...")
    recall_evaluator = EvidenceRecallEvaluator(pipeline)
    recall_results = recall_evaluator.evaluate_dev_set(dev_items, args.budget)
    recall_summary = recall_evaluator.summarize(recall_results)
    
    print(f"  Retrieval Recall:    {recall_summary.mean_recall_retrieval:.1%} (median: {recall_summary.median_recall_retrieval:.1%})")
    print(f"  Selection Recall:    {recall_summary.mean_recall_selection:.1%} (median: {recall_summary.median_recall_selection:.1%})")
    for qtype, m in recall_summary.per_type.items():
        print(f"    {qtype:30s} Sel: {m['mean_recall_selection']:.1%}  Ret: {m['mean_recall_retrieval']:.1%}  (n={m['count']})")
    
    # 8. Budget sweep (optional, fast)
    print("\nRunning budget sweep for recall curve...")
    budget_levels = [256, 512, 768, 1024, 1300, 1536, 2048, 3072]
    budget_sweep = {}
    for b in budget_levels:
        r = recall_evaluator.evaluate_dev_set(dev_items, b)
        s = recall_evaluator.summarize(r)
        budget_sweep[b] = s.mean_recall_selection
        print(f"  Budget {b:4d}: Selection Recall = {s.mean_recall_selection:.1%}")
    
    # 9. Statistical confidence intervals
    print("\nComputing bootstrap confidence intervals...")
    tester = StatisticalTester(n_bootstrap=args.n_bootstrap)
    acc_ci = tester.bootstrap_ci([float(p['is_correct']) for p in predictions], "accuracy")
    tok_ci = tester.bootstrap_ci([float(p['prompt_tokens']) for p in predictions], "tokens")
    
    print(f"  Accuracy 95% CI: [{acc_ci.ci_lower:.1%}, {acc_ci.ci_upper:.1%}]")
    print(f"  Tokens 95% CI:   [{tok_ci.ci_lower:.0f}, {tok_ci.ci_upper:.0f}]")
    
    # 10. Prepare output
    output = {
        'phase': 'phase0_baseline',
        'config': {
            'budget': args.budget,
            'use_embeddings': args.use_embeddings,
            'n_dev': len(dev_items),
        },
        'type_classifier': {
            'accuracy': cls_results['accuracy'],
            'per_type': cls_results['per_type'],
            'confusion_matrix': cls_results['confusion_matrix'],
        },
        'pipeline_metrics': metrics,
        'evidence_recall': {
            'mean_recall_retrieval': recall_summary.mean_recall_retrieval,
            'mean_recall_selection': recall_summary.mean_recall_selection,
            'median_recall_retrieval': recall_summary.median_recall_retrieval,
            'median_recall_selection': recall_summary.median_recall_selection,
            'per_type': recall_summary.per_type,
        },
        'budget_sweep': budget_sweep,
        'confidence_intervals': {
            'accuracy': {'mean': acc_ci.mean, 'ci_lower': acc_ci.ci_lower, 'ci_upper': acc_ci.ci_upper},
            'tokens': {'mean': tok_ci.mean, 'ci_lower': tok_ci.ci_lower, 'ci_upper': tok_ci.ci_upper},
        },
        'predictions': predictions,
        'elapsed_seconds': elapsed,
    }
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    print("=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())