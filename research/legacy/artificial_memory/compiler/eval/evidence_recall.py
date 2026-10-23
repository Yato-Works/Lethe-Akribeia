"""
Evidence Recall @ Budget Measurement for Lethe-Akribeia V4.

Evaluates retrieval + selection quality WITHOUT calling the LLM.
Measures: "How many gold evidence sessions survive the budgeted compiler?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler, CompiledContext
from artificial_memory.compiler.certificates.coprocessor_engine import CoProcessorBundle
from artificial_memory.compiler.graph.dependency_graph import EvidenceDependencyGraph
from artificial_memory.core.ir.structured import StructuredIR


@dataclass(frozen=True)
class EvidenceRecallResult:
    """Result of evidence recall measurement for one question."""
    question_id: str
    question_type: str
    gold_session_ids: set[str]
    retrieved_session_ids: set[str]
    selected_session_ids: set[str]  # After budget compilation
    recall_retrieval: float          # retrieved ∩ gold / gold
    recall_selection: float          # selected ∩ gold / gold
    n_gold: int
    n_retrieved: int
    n_selected: int
    budget_used: int
    budget_limit: int


@dataclass
class EvidenceRecallSummary:
    """Aggregate recall metrics across questions."""
    total_questions: int
    mean_recall_retrieval: float
    mean_recall_selection: float
    median_recall_retrieval: float
    median_recall_selection: float
    per_type: dict[str, dict[str, float]]
    budget_vs_recall: dict[int, float]  # budget -> mean recall


class EvidenceRecallEvaluator:
    """
    Fast evaluation of evidence recall at various budget levels.
    
    Does NOT call LLM - only measures retrieval + compilation pipeline.
    Runs in seconds for full Dev set.
    """
    
    def __init__(
        self,
        pipeline: CognitiveCompilerPipeline,
        budget_levels: list[int] | None = None,
    ):
        self.pipeline = pipeline
        self.budget_levels = budget_levels or [256, 512, 1024, 1536, 2048, 3072]
    
    def evaluate_question(
        self,
        question: str,
        gold_session_ids: set[str],
        haystack_sessions: list[list[dict]],
        haystack_dates: list[str],
        haystack_session_ids: list[str],
        budget: int,
    ) -> EvidenceRecallResult:
        """
        Evaluate recall for a single question at a specific budget.
        
        Args:
            question: The question text
            gold_session_ids: Session IDs that contain the answer (from benchmark)
            haystack_sessions: Full conversation history
            haystack_dates: Session dates
            haystack_session_ids: Session IDs
            budget: Token budget for compiler
            
        Returns:
            EvidenceRecallResult with recall metrics
        """
        # Run retrieval + compilation (no LLM answer generation)
        # We need to intercept the pipeline before the final LLM call
        
        # Build evidence dependency graph
        edg = EvidenceDependencyGraph()
        graph = edg.build(
            question=question,
            sessions=haystack_sessions,
            session_dates=haystack_dates,
            session_ids=haystack_session_ids,
        )
        
        # Run budgeted compiler
        compiler = BudgetedCognitiveCompiler(token_budget=budget)
        compilation_result = compiler.compile(graph, question)
        
        # Extract selected session IDs from compilation result
        selected_session_ids = set()
        for ev in compilation_result.selected_evidence:
            if hasattr(ev, 'session_id') and ev.session_id:
                selected_session_ids.add(ev.session_id)
            elif 'session_id' in ev.metadata:
                selected_session_ids.add(ev.metadata['session_id'])
        
        # For retrieval recall, we check what the graph builder retrieved
        retrieved_session_ids = set(graph.nodes.keys()) if hasattr(graph, 'nodes') else set()
        
        # Compute recalls
        n_gold = len(gold_session_ids)
        recall_retrieval = len(retrieved_session_ids & gold_session_ids) / n_gold if n_gold > 0 else 1.0
        recall_selection = len(selected_session_ids & gold_session_ids) / n_gold if n_gold > 0 else 1.0
        
        return EvidenceRecallResult(
            question_id="",  # Will be filled by caller
            question_type="",  # Will be filled by caller
            gold_session_ids=gold_session_ids,
            retrieved_session_ids=retrieved_session_ids,
            selected_session_ids=selected_session_ids,
            recall_retrieval=recall_retrieval,
            recall_selection=recall_selection,
            n_gold=n_gold,
            n_retrieved=len(retrieved_session_ids),
            n_selected=len(selected_session_ids),
            budget_used=compilation_result.total_tokens,
            budget_limit=budget,
        )
    
    def evaluate_dev_set(
        self,
        dev_items: list[dict],
        budget: int,
    ) -> list[EvidenceRecallResult]:
        """Evaluate recall on entire Dev set at one budget level."""
        results = []
        
        for item in dev_items:
            # Extract gold session IDs from answer_session_ids
            gold_sessions = set(item.get('answer_session_ids', []))
            
            result = self.evaluate_question(
                question=item['question'],
                gold_session_ids=gold_sessions,
                haystack_sessions=item['haystack_sessions'],
                haystack_dates=item['haystack_dates'],
                haystack_session_ids=item['haystack_session_ids'],
                budget=budget,
            )
            
            result = EvidenceRecallResult(
                question_id=item.get('question_id', ''),
                question_type=item.get('question_type', ''),
                gold_session_ids=result.gold_session_ids,
                retrieved_session_ids=result.retrieved_session_ids,
                selected_session_ids=result.selected_session_ids,
                recall_retrieval=result.recall_retrieval,
                recall_selection=result.recall_selection,
                n_gold=result.n_gold,
                n_retrieved=result.n_retrieved,
                n_selected=result.n_selected,
                budget_used=result.budget_used,
                budget_limit=result.budget_limit,
            )
            results.append(result)
        
        return results
    
    def evaluate_all_budgets(self, dev_items: list[dict]) -> dict[int, list[EvidenceRecallResult]]:
        """Evaluate at all budget levels."""
        return {b: self.evaluate_dev_set(dev_items, b) for b in self.budget_levels}
    
    def summarize(self, results: list[EvidenceRecallResult]) -> EvidenceRecallSummary:
        """Create summary statistics from results."""
        if not results:
            return EvidenceRecallSummary(
                total_questions=0,
                mean_recall_retrieval=0.0,
                mean_recall_selection=0.0,
                median_recall_retrieval=0.0,
                median_recall_selection=0.0,
                per_type={},
                budget_vs_recall={},
            )
        
        # Overall
        retrieval_recalls = [r.recall_retrieval for r in results]
        selection_recalls = [r.recall_selection for r in results]
        
        # Per-type
        per_type = {}
        for r in results:
            qtype = r.question_type
            if qtype not in per_type:
                per_type[qtype] = {'retrieval': [], 'selection': []}
            per_type[qtype]['retrieval'].append(r.recall_retrieval)
            per_type[qtype]['selection'].append(r.recall_selection)
        
        per_type_summary = {}
        for qtype, vals in per_type.items():
            per_type_summary[qtype] = {
                'mean_recall_retrieval': float(np.mean(vals['retrieval'])),
                'mean_recall_selection': float(np.mean(vals['selection'])),
                'median_recall_retrieval': float(np.median(vals['retrieval'])),
                'median_recall_selection': float(np.median(vals['selection'])),
                'count': len(vals['retrieval']),
            }
        
        return EvidenceRecallSummary(
            total_questions=len(results),
            mean_recall_retrieval=float(np.mean(retrieval_recalls)),
            mean_recall_selection=float(np.mean(selection_recalls)),
            median_recall_retrieval=float(np.median(retrieval_recalls)),
            median_recall_selection=float(np.median(selection_recalls)),
            per_type=per_type_summary,
            budget_vs_recall={},
        )
    
    def budget_sweep_summary(self, all_results: dict[int, list[EvidenceRecallResult]]) -> dict[int, float]:
        """Create budget vs recall curve."""
        return {
            budget: self.summarize(res).mean_recall_selection
            for budget, res in all_results.items()
        }


def load_dev_split(dev_path: str | Path) -> list[dict]:
    """Load Dev split with full haystack data."""
    path = Path(dev_path)
    
    with open(path) as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        items = data
    elif 'results' in data:
        items = data['results']
    elif 'items' in data:
        items = data['items']
    else:
        items = [data]
    
    # Normalize
    normalized = []
    for item in items:
        normalized.append({
            'question_id': item.get('question_id', item.get('id', '')),
            'question': item.get('question', ''),
            'question_type': item.get('question_type', item.get('type', 'unknown')),
            'answer_session_ids': item.get('answer_session_ids', item.get('evidence_sessions', [])),
            'haystack_sessions': item.get('haystack_sessions', item.get('sessions', [])),
            'haystack_dates': item.get('haystack_dates', item.get('dates', [])),
            'haystack_session_ids': item.get('haystack_session_ids', item.get('session_ids', [])),
        })
    
    return normalized


if __name__ == "__main__":
    # Quick test structure
    print("EvidenceRecallEvaluator - Ready for integration")
    print("Usage:")
    print("  evaluator = EvidenceRecallEvaluator(pipeline)")
    print("  results = evaluator.evaluate_dev_set(dev_items, budget=1300)")
    print("  summary = evaluator.summarize(results)")
    print("  print(f'Selection Recall: {summary.mean_recall_selection:.1%}')")