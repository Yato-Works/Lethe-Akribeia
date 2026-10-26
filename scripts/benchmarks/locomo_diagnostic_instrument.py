"""LoCoMo Diagnostic Instrument Engine.

Implements rigorous failure attribution & evidence metrics:
- Evidence Recall: Did Candidate Retrieval find all gold evidence turns?
- Evidence Coverage / Admission: Did Budgeted Compilation retain the evidence in context?
- Failure Attribution:
    - SUCCESS: Model answered correctly.
    - READER_FAILURE: 100% of gold evidence was present in context, yet Reader failed (Reader's fault!).
    - KNAPSACK_DROP: Evidence was retrieved by V4, but dropped during knapsack/compiler (Compiler's fault!).
    - RETRIEVAL_MISS: Evidence was never retrieved in top candidates (Retriever's fault!).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_locomo_full import (
    load_locomo,
    conversation_turns,
    load_questions,
    score,
    _enumeration_probe_turns,
    _scaled_budget,
)
from collections import Counter, defaultdict
from eval_v4_pipeline_dev import V4CandidateRetriever
from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.recall.fact_index import FactIndex


def run_diagnostic(limit: int = 50, run_json_path: Path | None = None) -> dict:
    data = load_locomo(REPO / "datasets" / "external" / "locomo10.json")
    conv_data = data[0]
    questions = load_questions(conv_data)[:limit]

    # Map dialogue ID -> text, and text -> dialogue ID
    turns, dia_map = conversation_turns(conv_data["conversation"])
    text_to_did = {text.strip(): did for did, text in dia_map.items()}

    # Also build normalised lookup for robust matching
    def norm_clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    norm_to_did = {norm_clean(text): did for did, text in dia_map.items()}

    # Load predictions if existing run provided, else we can re-evaluate
    preds_by_qid = {}
    if run_json_path and run_json_path.exists():
        with open(run_json_path, encoding="utf-8") as f:
            prev_run = json.load(f)
            for r in prev_run["results"]:
                preds_by_qid[r["question_id"]] = (r["predicted_answer"], r["is_correct"])

    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    planner = CognitiveExecutionPlanner()
    compiler = BudgetedCognitiveCompiler()
    graph_builder = GraphBuilder()
    fact_idx = FactIndex()
    fact_idx.build(turns)

    diagnostics = []
    
    total_q = len(questions)
    retrieval_complete_count = 0
    context_complete_count = 0
    correct_count = 0

    attribution_counts = Counter()

    print("=" * 80)
    print(f"RUNNING LOCOMO DIAGNOSTIC INSTRUMENT ON {total_q} QUESTIONS")
    print("=" * 80)

    for idx, q in enumerate(questions):
        qid = q["question_id"]
        question_text = q["question"]
        gold_answer = q["answer"]
        gold_evidence_dids = q.get("evidence", [])
        cat = q["category"]

        # Run pipeline stages
        plan = planner.plan(question_text, [], reference_date=None)
        token_budget = plan.recommended_budget

        top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
            query=question_text, all_turns=turns, retriever=retriever,
            reference_date=None, base_top_k=45)

        base_limit = 32 if plan.dominant_mode == "aggregation" else 28
        top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
        cand = [t for t in turns if t[0] in top_sids]
        if not cand:
            cand = list(top_turns[:25])

        # FactIndex recall
        fact_turns = fact_idx.query(question_text, turns, max_turns=16)
        already_cand = {(t[0], t[3]) for t in cand}
        for ft in fact_turns:
            key = (ft[0], ft[3])
            if key not in already_cand:
                cand.append(ft)
                already_cand.add(key)
            turn_scores[key] = max(turn_scores.get(key, 0.0), 25.0)
            if ft not in top_turns:
                top_turns.append(ft)

        # Enumeration probe
        _probe = _enumeration_probe_turns(
            question_text, turns, retriever,
            already={(t[0], t[3]) for t in cand})
        if _probe:
            cand = cand + _probe

        # Budget calibration
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
            question=question_text, bundles=bundles,
            token_budget=token_budget, anchor_scores=anchor_scores,
            is_multi_session=len(top_sids) > 1, adaptive_escalation=True)

        context_text = compiled.context_text
        selected_bids = set(compiled.selected_bundle_ids)

        # 1. Measure Retrieval Recall
        # Check which gold dids appear in top_turns
        retrieved_dids = set()
        for t in top_turns:
            did = text_to_did.get(t[3].strip()) or norm_to_did.get(norm_clean(t[3]))
            if did:
                retrieved_dids.add(did)

        gold_set = set(gold_evidence_dids)
        retrieved_gold = gold_set & retrieved_dids
        retrieval_recall = len(retrieved_gold) / len(gold_set) if gold_set else 1.0
        retrieval_complete = (retrieved_gold == gold_set) if gold_set else True
        if retrieval_complete:
            retrieval_complete_count += 1

        # 2. Measure Knapsack Admission / Context Coverage
        # Check which gold dids appear in compiled context (via bundle selection or clean substring)
        context_dids = set()
        clean_ctx = norm_clean(context_text)
        for did in gold_evidence_dids:
            ev_text = dia_map.get(did, "").strip()
            if not ev_text:
                continue
            # A: Check via bundle inclusion
            in_selected_bundle = False
            for b in bundles:
                if b.bundle_id in selected_bids:
                    if any(ev_text in n.raw_text or n.raw_text in ev_text for n in b.member_nodes):
                        in_selected_bundle = True
                        break
            # B: Check via clean substring
            clean_ev = norm_clean(ev_text)
            if in_selected_bundle or (len(clean_ev) >= 20 and clean_ev[:35] in clean_ctx):
                context_dids.add(did)

        context_gold = gold_set & context_dids
        context_coverage = len(context_gold) / len(gold_set) if gold_set else 1.0
        context_complete = (context_gold == gold_set) if gold_set else True
        if context_complete:
            context_complete_count += 1

        # Prediction and correctness
        if qid in preds_by_qid:
            pred_ans, is_ok = preds_by_qid[qid]
        else:
            pred_ans = ""
            is_ok = False

        # Determine exclusive terminal state and semantic sufficiency
        # Note: Stage metrics (retrieval_complete, context_complete) are tracked separately.
        # Exclusive terminal state partitions all 50 questions with sum = 50.
        if is_ok:
            correct_count += 1
            if context_complete:
                term_state = "SUCCESS_FULL_CONTEXT"
            else:
                term_state = "SUCCESS_PARTIAL_CONTEXT"
            attr = "SUCCESS"
            semantic_sufficient = True
        else:
            if qid == "lc_conv-26_23":
                # Gold answer relies on image attachment ("Nothing is Impossible") absent from text corpus
                term_state = "MODALITY_DEFICIENT"
                attr = "MODALITY_DEFICIENT"
                semantic_sufficient = False
            elif context_complete:
                # All gold evidence text was present in Reader input, yet Reader failed
                term_state = "READER_FAILURE"
                attr = "READER_FAILURE"
                semantic_sufficient = True
            elif retrieval_complete:
                # Evidence was retrieved in candidate set, but dropped by budgeted compiler
                term_state = "KNAPSACK_DROP"
                attr = "KNAPSACK_DROP"
                semantic_sufficient = False
            else:
                # Evidence was never fully retrieved in candidate set
                term_state = "RETRIEVAL_MISS"
                attr = "RETRIEVAL_MISS"
                semantic_sufficient = False

        attribution_counts[term_state] += 1

        diagnostics.append({
            "question_id": qid,
            "category": cat,
            "question": question_text,
            "gold_answer": gold_answer,
            "predicted_answer": pred_ans,
            "is_correct": is_ok,
            "gold_evidence": list(gold_set),
            "retrieved_gold": list(retrieved_gold),
            "context_gold": list(context_gold),
            "retrieval_recall": retrieval_recall,
            "context_coverage": context_coverage,
            "retrieval_complete": retrieval_complete,
            "context_complete": context_complete,
            "semantic_sufficient": semantic_sufficient,
            "exclusive_terminal_state": term_state,
            "attribution": attr,
        })

    # Cross tabulations
    crosstab_s_x_r = defaultdict(list)
    crosstab_s_x_c = defaultdict(list)
    crosstab_r_x_c = defaultdict(list)
    for d in diagnostics:
        qid_item = d["question_id"]
        crosstab_s_x_r[(d["is_correct"], d["retrieval_complete"])].append(qid_item)
        crosstab_s_x_c[(d["is_correct"], d["context_complete"])].append(qid_item)
        crosstab_r_x_c[(d["retrieval_complete"], d["context_complete"])].append(qid_item)

    print(f"\nDIAGNOSTIC SUMMARY ({total_q} questions):")
    print(f"  Accuracy:                    {correct_count}/{total_q} ({correct_count/total_q*100:.1f}%)")
    print(f"  Retrieval Complete (Recall): {retrieval_complete_count}/{total_q} ({retrieval_complete_count/total_q*100:.1f}%)")
    print(f"  Context Complete (Coverage): {context_complete_count}/{total_q} ({context_complete_count/total_q*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("CROSS-TABULATION 1: SUCCESS x RETRIEVAL_COMPLETE")
    print("=" * 60)
    for (s, r), qids in sorted(crosstab_s_x_r.items()):
        print(f"  SUCCESS={str(s):<5} | RETRIEVAL_COMPLETE={str(r):<5} : {len(qids):2d} questions")

    print("\n" + "=" * 60)
    print("CROSS-TABULATION 2: SUCCESS x CONTEXT_COMPLETE")
    print("=" * 60)
    for (s, c), qids in sorted(crosstab_s_x_c.items()):
        print(f"  SUCCESS={str(s):<5} | CONTEXT_COMPLETE={str(c):<5}   : {len(qids):2d} questions")

    print("\n" + "=" * 60)
    print("CROSS-TABULATION 3: RETRIEVAL_COMPLETE x CONTEXT_COMPLETE")
    print("=" * 60)
    for (r, c), qids in sorted(crosstab_r_x_c.items()):
        print(f"  RETR_COMP={str(r):<5} | CTX_COMP={str(c):<5}        : {len(qids):2d} questions")

    print("\n" + "=" * 60)
    print("EXCLUSIVE TERMINAL STATES (SUM = 50):")
    print("=" * 60)
    for attr, count in attribution_counts.most_common():
        pct = count / total_q * 100
        print(f"  {attr:24s}: {count:2d} ({pct:5.1f}%)")

    return {
        "summary": {
            "total": total_q,
            "correct": correct_count,
            "retrieval_complete": retrieval_complete_count,
            "context_complete": context_complete_count,
            "exclusive_terminal_states": dict(attribution_counts),
            "crosstabs": {
                "success_x_retrieval": {f"S_{s}_R_{r}": qids for (s, r), qids in crosstab_s_x_r.items()},
                "success_x_context": {f"S_{s}_C_{c}": qids for (s, c), qids in crosstab_s_x_c.items()},
                "retrieval_x_context": {f"R_{r}_C_{c}": qids for (r, c), qids in crosstab_r_x_c.items()},
            },
        },
        "details": diagnostics,
    }


if __name__ == "__main__":
    from collections import Counter
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        res_path = Path(sys.argv[1])
    else:
        # Pick latest timestamped json
        jsons = sorted((REPO / "benchmark_results").glob("locomo_full_*.json"))
        if jsons:
            res_path = jsons[-1]
        else:
            res_path = REPO / "benchmark_results" / "locomo_full_20261011_014224.json"
    print(f"Loading predictions from: {res_path}")
    diag = run_diagnostic(50, res_path)
    out_file = REPO / "benchmark_results" / "locomo_diagnostic_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full diagnostic report to -> {out_file}")
