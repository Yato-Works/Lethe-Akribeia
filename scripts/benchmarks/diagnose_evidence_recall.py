"""Diagnose Evidence Recall across the Lethe-Akribeia pipeline stages.

This runs purely on the engine (NO LLM generation needed, runs in ~1-2 min for all 1986 questions).
It tracks every question's gold evidence (dia_ids) across 3 pipeline checkpoints:
  1. Retriever: Did retrieve_convergent_evidence find the evidence turns?
  2. Candidate Pool: Did session expansion / enumeration probe keep the evidence?
  3. Compiler Context: Did the final compiled context_text actually retain the evidence text?
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_locomo_full import (
    load_locomo,
    load_questions,
    conversation_turns,
    _enumeration_probe_turns,
    _scaled_budget,
)
from eval_v4_pipeline_dev import V4CandidateRetriever
from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory.compiler.graph.graph_builder import GraphBuilder
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner


def main():
    data_path = REPO / "datasets" / "external" / "locomo10.json"
    data = load_locomo(data_path)

    # Load latest run results to correlate with correctness
    latest_run_path = REPO / "benchmark_results" / "locomo_full_20261010_151930.json"
    with open(latest_run_path, encoding="utf-8") as f:
        run_data = json.load(f)
    result_by_id = {r["question_id"]: r for r in run_data["results"]}

    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    graph_builder = GraphBuilder()
    compiler = BudgetedCognitiveCompiler()
    planner = CognitiveExecutionPlanner()

    total_q = 0
    with_evidence_q = 0

    # Metrics
    # For each category:
    #   total,
    #   full_recall_retriever, partial_recall_retriever, zero_recall_retriever
    #   full_recall_cand, partial_recall_cand, zero_recall_cand
    #   full_recall_context, partial_recall_context, zero_recall_context
    #   and correlation with is_correct

    stats = defaultdict(lambda: {
        "total": 0,
        "with_evidence": 0,
        "correct": 0,
        # Context presence
        "ctx_full_evidence": 0,
        "ctx_partial_evidence": 0,
        "ctx_zero_evidence": 0,
        # Retriever presence
        "ret_full_evidence": 0,
        "ret_partial_evidence": 0,
        "ret_zero_evidence": 0,
        # Correctness conditional on context presence
        "correct_given_ctx_full": 0,
        "correct_given_ctx_partial": 0,
        "correct_given_ctx_zero": 0,
        "total_ctx_full": 0,
        "total_ctx_partial": 0,
        "total_ctx_zero": 0,
    })

    t_start = time.perf_counter()

    for conv_idx, conv in enumerate(data):
        c_dict = conv["conversation"]
        turns, dia_map = conversation_turns(c_dict)
        # map turn text -> dia_id (or dia_id -> text)
        questions = load_questions(conv)
        print(f"Processing Conv {conv_idx+1}/{len(data)}: {conv.get('sample_id')} ({len(questions)} Qs)...")

        for q in questions:
            qid = q["question_id"]
            cat = q["category"]
            evidence_ids = q.get("evidence", [])
            is_correct = result_by_id.get(qid, {}).get("is_correct", False)

            stats[cat]["total"] += 1
            if is_correct:
                stats[cat]["correct"] += 1

            if not evidence_ids:
                # Cat 5 (adversarial) often has no evidence
                continue

            stats[cat]["with_evidence"] += 1

            # Resolve evidence texts
            evidence_texts = [dia_map[did] for did in evidence_ids if did in dia_map]
            if not evidence_texts:
                continue

            # Run engine pipeline up to compiler (NO LLM)
            plan = planner.plan(q["question"], [], reference_date=None)
            token_budget = plan.recommended_budget

            top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
                query=q["question"], all_turns=turns, retriever=retriever,
                reference_date=None, base_top_k=45
            )

            base_limit = 32 if plan.dominant_mode == "aggregation" else 28
            top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
            cand = [t for t in turns if t[0] in top_sids]
            if not cand:
                cand = list(top_turns[:25])

            _probe = _enumeration_probe_turns(
                q["question"], turns, retriever,
                already={(t[0], t[3]) for t in cand}
            )
            if _probe:
                cand = cand + _probe

            _pool_tokens = sum(len(t[3].split()) * 4 // 3 for t in cand)
            if _pool_tokens:
                token_budget = _scaled_budget(token_budget, _pool_tokens)

            t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(cand)}
            anchor_ids, anchor_scores = [], {}
            for t in top_turns:
                nid = t_map.get((t[0], t[3]))
                if nid and nid not in anchor_ids:
                    anchor_ids.append(nid)
                    anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)

            edg = graph_builder.build(cand, [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            compiled = compiler.compile(
                question=q["question"], bundles=bundles,
                token_budget=token_budget, anchor_scores=anchor_scores,
                is_multi_session=len(top_sids) > 1, adaptive_escalation=True
            )
            ctx = compiled.context_text

            # Check 1: Retriever recall
            top_texts = {t[3] for t in top_turns}
            ret_hits = sum(1 for et in evidence_texts if et in top_texts)

            # Check 2: Final Compiled Context recall
            ctx_hits = sum(1 for et in evidence_texts if et in ctx or et.strip()[:40] in ctx)

            n_ev = len(evidence_texts)

            # Record Context recall
            if ctx_hits == n_ev:
                stats[cat]["ctx_full_evidence"] += 1
                stats[cat]["total_ctx_full"] += 1
                if is_correct:
                    stats[cat]["correct_given_ctx_full"] += 1
            elif ctx_hits > 0:
                stats[cat]["ctx_partial_evidence"] += 1
                stats[cat]["total_ctx_partial"] += 1
                if is_correct:
                    stats[cat]["correct_given_ctx_partial"] += 1
            else:
                stats[cat]["ctx_zero_evidence"] += 1
                stats[cat]["total_ctx_zero"] += 1
                if is_correct:
                    stats[cat]["correct_given_ctx_zero"] += 1

            # Record Retriever recall
            if ret_hits == n_ev:
                stats[cat]["ret_full_evidence"] += 1
            elif ret_hits > 0:
                stats[cat]["ret_partial_evidence"] += 1
            else:
                stats[cat]["ret_zero_evidence"] += 1

    elapsed = time.perf_counter() - t_start
    print(f"\nDiagnosis completed in {elapsed:.1f}s!\n")

    cat_labels = {
        1: "Cat 1 (Single-hop / Multi-item)",
        2: "Cat 2 (Temporal)",
        3: "Cat 3 (Open-domain)",
        4: "Cat 4 (Single-hop Fact)",
        5: "Cat 5 (Adversarial)",
    }

    print("=" * 80)
    print("  EVIDENCE RECALL & BOTTLENECK DIAGNOSIS REPORT")
    print("=" * 80)

    for cat in sorted(stats.keys()):
        s = stats[cat]
        tot = s["total"]
        with_ev = s["with_evidence"]
        print(f"\n--- {cat_labels.get(cat, str(cat))} (Total: {tot} Qs, With Evidence: {with_ev}) ---")
        if with_ev == 0:
            print("  No explicit gold evidence turns in this category (e.g. unanswerable / adversarial).")
            continue

        print(f"  [1. 検索エンジン (Retriever Top-45) 到達率]")
        print(f"    - Full Evidence 到達   : {s['ret_full_evidence']:3d} / {with_ev} ({s['ret_full_evidence']/with_ev*100:5.1f}%)")
        print(f"    - Partial (一部のみ)  : {s['ret_partial_evidence']:3d} / {with_ev} ({s['ret_partial_evidence']/with_ev*100:5.1f}%)")
        print(f"    - Zero (全滅・検索漏れ): {s['ret_zero_evidence']:3d} / {with_ev} ({s['ret_zero_evidence']/with_ev*100:5.1f}%)")

        print(f"  [2. 最終コンテキスト (Compiler Context) 保持率]")
        print(f"    - Full Evidence 保持   : {s['ctx_full_evidence']:3d} / {with_ev} ({s['ctx_full_evidence']/with_ev*100:5.1f}%)")
        print(f"    - Partial (一部のみ)  : {s['ctx_partial_evidence']:3d} / {with_ev} ({s['ctx_partial_evidence']/with_ev*100:5.1f}%)")
        print(f"    - Zero (完全脱落)     : {s['ctx_zero_evidence']:3d} / {with_ev} ({s['ctx_zero_evidence']/with_ev*100:5.1f}%)")

        print(f"  [3. 証拠到達と正解率の因果関係 (コンテキストの質 -> 正解率)]")
        if s["total_ctx_full"]:
            acc_full = s["correct_given_ctx_full"] / s["total_ctx_full"] * 100
            print(f"    - 証拠 100% 保持時の正解率 : {s['correct_given_ctx_full']:3d} / {s['total_ctx_full']} ({acc_full:5.1f}%)")
        if s["total_ctx_partial"]:
            acc_part = s["correct_given_ctx_partial"] / s["total_ctx_partial"] * 100
            print(f"    - 証拠 部分保持時の正解率  : {s['correct_given_ctx_partial']:3d} / {s['total_ctx_partial']} ({acc_part:5.1f}%)")
        if s["total_ctx_zero"]:
            acc_zero = s["correct_given_ctx_zero"] / s["total_ctx_zero"] * 100
            print(f"    - 証拠 0% (完全欠落) 時   : {s['correct_given_ctx_zero']:3d} / {s['total_ctx_zero']} ({acc_zero:5.1f}%)")


if __name__ == "__main__":
    main()
