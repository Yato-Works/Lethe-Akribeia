import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

from eval_locomo_full import (
    load_locomo,
    conversation_turns,
    load_questions,
    _scaled_budget,
)
from eval_v4_pipeline_dev import V4CandidateRetriever
from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory.compiler.graph.graph_builder import GraphBuilder
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner

data = load_locomo(REPO / "datasets" / "external" / "locomo10.json")
conv_data = data[0]
turns, dia_map = conversation_turns(conv_data["conversation"])
questions = load_questions(conv_data)

target_qids = ["lc_conv-26_10", "lc_conv-26_18", "lc_conv-26_22", "lc_conv-26_23", "lc_conv-26_47"]

retriever = V4CandidateRetriever()
coverage = MultiSessionCoverageEngine()
planner = CognitiveExecutionPlanner()
compiler = BudgetedCognitiveCompiler()
graph_builder = GraphBuilder()

print("="*80)
print("DEEP TRACE OF 5 KNAPSACK-DROPPED QUESTIONS")
print("="*80)

for q in questions:
    if q["question_id"] not in target_qids:
        continue
    qid = q["question_id"]
    q_text = q["question"]
    gold_dids = q.get("evidence", [])
    gold_texts = {did: dia_map.get(did, "") for did in gold_dids}
    
    plan = planner.plan(q_text, [], reference_date=None)
    token_budget = plan.recommended_budget

    top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
        query=q_text, all_turns=turns, retriever=retriever,
        reference_date=None, base_top_k=45)

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
        question=q_text, bundles=bundles,
        token_budget=token_budget, anchor_scores=anchor_scores,
        is_multi_session=len(top_sids) > 1, adaptive_escalation=True)

    print(f"\n[{qid}] {q_text}")
    print(f"  Gold Evidence: {gold_dids}")
    print(f"  Budget: {token_budget} tokens | Context total tokens: {compiled.total_tokens}")
    print(f"  Candidate turns count: {len(cand)} | Bundles count: {len(bundles)} | Selected bundles: {len(compiled.selected_bundle_ids)}")

    for did in gold_dids:
        gt_t = dia_map.get(did, "")
        in_all_turns = any(gt_t.strip() == t[3].strip() for t in turns)
        top_turn_ranks = [i+1 for i, t in enumerate(top_turns) if gt_t.strip() in t[3].strip() or t[3].strip() in gt_t.strip()]
        in_cand = any(gt_t.strip() in t[3].strip() for t in cand)
        
        # Check in bundles
        bundle_hits = [b.bundle_id for b in bundles if any(gt_t.strip() in n.raw_text.strip() for n in b.member_nodes)]
        # Check in selected bundles
        selected_hits = [b_id for b_id in compiled.selected_bundle_ids if b_id in bundle_hits]
        # Check in final context
        in_context = gt_t.strip()[:50] in compiled.context_text

        print(f"    - {did}: in_top_turns={top_turn_ranks}, in_cand={in_cand}, in_bundles={bundle_hits}, in_selected={selected_hits}, IN_CONTEXT={in_context}")
        if bundle_hits:
            for b_id in bundle_hits:
                b_obj = next(b for b in bundles if b.bundle_id == b_id)
                print(f"        Bundle {b_id}: tokens={b_obj.total_tokens}, primary_node_id={b_obj.primary_node_id}, anchor_score={anchor_scores.get(b_obj.primary_node_id, 0.0):.3f}")
