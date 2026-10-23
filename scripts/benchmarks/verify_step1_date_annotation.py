"""Verify the effect of annotate_context_relative_dates on Cat 2 failures using actual 7B model."""

from __future__ import annotations

import json
import re
import sys
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
    build_reader_prompt,
    score,
)
from eval_v4_pipeline_dev import V4CandidateRetriever
from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory.research.benchmarks.llm import OllamaAnswerer
from test_context_date_annotator import annotate_context_relative_dates


def main():
    data_path = REPO / "datasets" / "external" / "locomo10.json"
    data = load_locomo(data_path)

    latest_run_path = REPO / "benchmark_results" / "locomo_full_20261010_151930.json"
    with open(latest_run_path, encoding="utf-8") as f:
        run_data = json.load(f)
    result_by_id = {r["question_id"]: r for r in run_data["results"]}

    # Target test questions: 10 failing Cat 2 questions with relative date miscalculations
    target_qids = [
        "lc_conv-30_0", # When Jon has lost his job as a banker? (GT: 19 January, 2023)
        "lc_conv-30_1", # When Gina has lost her job at Door Dash? (GT: January, 2023)
        "lc_conv-30_6", # When did Jon go to a fair to get more exposure for his dance studio? (GT: 24 April, 2023)
        "lc_conv-30_12", # When did Jon visit networking events for his store? (GT: 20 June, 2023)
        "lc_conv-26_26", # When did Melanie read the book "nothing is impossible"? (GT: 2022)
        "lc_conv-26_31", # When did Melanie go camping in June? (GT: The week before 27 June 2023)
        "lc_conv-26_49", # When did Caroline and Melanie go to a pride fesetival together? (GT: 2022)
        "lc_conv-41_0",  # Sample from conv-41
        "lc_conv-41_4",
        "lc_conv-42_0",
    ]

    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    graph_builder = GraphBuilder()
    compiler = BudgetedCognitiveCompiler()
    planner = CognitiveExecutionPlanner()
    answerer = OllamaAnswerer(num_ctx=8192)

    tested = 0
    rescued = 0

    print("=" * 70)
    print("  VERIFYING STEP 1: CONTEXT RELATIVE DATE ANNOTATION")
    print("=" * 70)

    for conv in data:
        c_dict = conv["conversation"]
        turns, dia_map = conversation_turns(c_dict)
        questions = load_questions(conv)

        for q in questions:
            qid = q["question_id"]
            if qid not in target_qids:
                continue

            tested += 1
            prev_pred = result_by_id.get(qid, {}).get("predicted_answer", "N/A")
            prev_ok = result_by_id.get(qid, {}).get("is_correct", False)
            gt = q["answer"]

            # Run compilation
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

            # APPLY STEP 1: Pre-normalize relative dates in compiled context
            annotated_ctx = annotate_context_relative_dates(compiled.context_text)

            prompt = build_reader_prompt(q["question"], annotated_ctx)
            new_pred = answerer.answer(q["question"], prompt).text.strip()
            if new_pred.endswith(".") and len(new_pred.split()) <= 2:
                new_pred = new_pred[:-1].strip()

            new_ok = score(new_pred, gt)
            if new_ok and not prev_ok:
                rescued += 1
                status = "RESCUED (0 -> 1) !"
            elif new_ok and prev_ok:
                status = "STILL CORRECT"
            elif not new_ok and not prev_ok:
                status = "STILL FAILING"
            else:
                status = "REGRESSED (1 -> 0)"

            print(f"\n[{qid}] {status}")
            print(f"  Q   : {q['question']}")
            print(f"  GT  : {gt}")
            print(f"  Old : {prev_pred} (ok={prev_ok})")
            print(f"  New : {new_pred} (ok={new_ok})")

    print(f"\nSummary: Rescued {rescued}/{tested} questions tested!")


if __name__ == "__main__":
    main()
