"""Gate 1 Evaluation Runner: Integrated Evidence Closure Engine (ECE) Controller.

Integrates:
1. Deterministic Query Planning (Write LLM = 0).
2. Multi-Session Coverage Pool + FactIndex Multi-Hop Coreference (Sweden, etc.).
3. ECE Closure Engine verifying evidence requirements (SATISFIED, PARTIAL, CONFLICTING, UNRESOLVED).
4. Dialogue Dependency Graph bundling preserving conversational turn pairs.
5. Dynamic Marginal UtilityKnapsack compilation.
6. Frozen Reader Prompt & Deterministic Relative-Date Resolution.

Strict Contract:
- Frozen Ollama Reader parameters (qwen2.5:7b-instruct, temp=0.0, seed=42).
- Zero gold contamination or runtime question leaks.
- Independent output destination: benchmark_results/experiment_ece_gate1/
"""

from __future__ import annotations

import argparse
import datetime
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
    conversation_turns,
    load_questions,
    score,
    resolve_relative_dates,
    build_reader_prompt,
    _scaled_budget,
    _enumeration_probe_turns,
    CAT_NAMES,
)
from eval_v4_pipeline_dev import V4CandidateRetriever
from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory.compiler.graph.graph_builder import GraphBuilder
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory.recall.fact_index import FactIndex
from artificial_memory.compiler.closure import (
    DeterministicQueryPlanner,
    EvidenceClosureEngine,
    AnswerTopology,
)
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


def run_gate1(args) -> None:
    data = load_locomo(Path(args.data))
    jobs: list[tuple[dict, dict]] = []
    for conv in data:
        for q in load_questions(conv):
            jobs.append((conv, q))
    if args.limit:
        jobs = jobs[: args.limit]

    print("=" * 70)
    print("  Evidence Closure Engine (ECE) / Memory Compiler v2 - Gate 1 Audit")
    print(f"  Evaluating {len(jobs)} questions on frozen Ollama Reader")
    print("=" * 70)

    # Core Components
    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    graph_builder = GraphBuilder()
    compiler = BudgetedCognitiveCompiler()
    planner = DeterministicQueryPlanner()
    closure_engine = EvidenceClosureEngine()
    answerer = OllamaAnswerer(num_ctx=args.num_ctx)

    results: list[dict] = []
    by_cat: dict[int, list[bool]] = defaultdict(list)
    t_start = time.perf_counter()

    turn_cache: dict[str, list] = {}
    fact_cache: dict[str, FactIndex] = {}

    for idx, (conv, q) in enumerate(jobs):
        sid = conv.get("sample_id", "x")
        if sid not in turn_cache:
            turn_cache[sid] = conversation_turns(conv["conversation"])
        all_turns, dia_map = turn_cache[sid]

        if sid not in fact_cache:
            fidx = FactIndex()
            fidx.build(all_turns)
            fact_cache[sid] = fidx
        fact_idx = fact_cache[sid]

        t0 = time.perf_counter()
        try:
            # 1. Deterministic Query Planning (Write LLM = 0)
            query_plan = planner.plan(q["question"])

            # 2. Rich Multi-Session Candidate Pool
            top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
                query=q["question"], all_turns=all_turns, retriever=retriever,
                reference_date=None, base_top_k=45)

            base_limit = 32 if query_plan.topology == AnswerTopology.SET_ENUMERATION else 28
            top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
            cand = [t for t in all_turns if t[0] in top_sids]
            if not cand:
                cand = list(top_turns[:25])

            # FactIndex multi-hop coreference retrieval (e.g. "my home country" -> "Sweden")
            fact_turns = fact_idx.query(q["question"], all_turns, max_turns=16)
            already_cand = {(t[0], t[3]) for t in cand}
            for ft in fact_turns:
                key = (ft[0], ft[3])
                if key not in already_cand:
                    cand.append(ft)
                    already_cand.add(key)
                turn_scores[key] = max(turn_scores.get(key, 0.0), 25.0)
                if ft not in top_turns:
                    top_turns.append(ft)

            # Plural head probe turns for enumeration
            _probe = _enumeration_probe_turns(
                q["question"], all_turns, retriever,
                already={(t[0], t[3]) for t in cand})
            if _probe:
                cand = cand + _probe

            # 3. ECE Closure Loop over the candidate pool
            closure_res = closure_engine.run_closure(
                query_plan=query_plan,
                all_turns=cand,
                initial_top_k=35,
            )

            # 4. Inject ECE Closure Verified Bonuses into Anchor Scores
            t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(cand)}
            anchor_ids = []
            anchor_scores = {}

            # Baseline anchor seeds from top_turns
            for t in top_turns:
                nid = t_map.get((t[0], t[3]))
                if nid and nid not in anchor_ids:
                    anchor_ids.append(nid)
                    anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)

            # Ensure anchors that specifically fulfill ECE requirements are reachable
            for req in closure_res.requirement_details:
                for stid in req.supporting_turn_ids:
                    for c in closure_res.admitted_candidates:
                        if c.turn_id == stid:
                            nid = t_map.get((c.session_id, c.content))
                            if nid and nid not in anchor_ids:
                                anchor_ids.append(nid)
                                anchor_scores[nid] = turn_scores.get((c.session_id, c.content), 2.0)

            # 5. Scaled token budget
            _pool_tokens = sum(len(t[3].split()) * 4 // 3 for t in cand)
            token_budget = _scaled_budget(query_plan.recommended_budget, _pool_tokens)
            if query_plan.topology == AnswerTopology.INFERENCE_BOOLEAN:
                # Keep inference context tight to prevent irrelevant chatter from distracting reader
                token_budget = min(1200, token_budget)

            # 6. Dialogue Dependency Graph Bundling & Compilation
            edg = graph_builder.build(cand, [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            compiled = compiler.compile(
                question=q["question"], bundles=bundles,
                token_budget=token_budget, anchor_scores=anchor_scores,
                is_multi_session=len(top_sids) > 1, adaptive_escalation=True)

            # 7. Frozen Reader Prompt & Inference
            prompt_text = build_reader_prompt(q["question"], compiled.context_text)
            pred = answerer.answer(q["question"], prompt_text).text.strip()

            if pred.endswith(".") and len(pred.split()) <= 2:
                pred = pred[:-1].strip()

            # 8. Deterministic Relative Date Resolution
            if re.search(r"\bwhen\b", q["question"], re.I):
                pred = resolve_relative_dates(pred, compiled.context_text, q["question"])

            ok = score(pred, q["answer"])
            duration = time.perf_counter() - t0

            q_record = {
                "question_id": q["question_id"],
                "category": q["category"],
                "question": q["question"],
                "ground_truth": q["answer"],
                "predicted_answer": pred,
                "is_correct": ok,
                "topology": query_plan.topology.value,
                "closure_completeness": closure_res.closure_completeness,
                "satisfied_requirements": list(sorted(closure_res.closure_state.satisfied_requirements)),
                "unmet_requirements": list(closure_res.unmet_requirements),
                "conflicting_requirements": list(sorted(closure_res.closure_state.conflicting_requirements)),
                "termination_reason": closure_res.closure_state.termination_reason,
                "iterations_run": closure_res.closure_state.iteration,
                "context_tokens": compiled.total_tokens,
                "latency_seconds": round(duration, 3),
            }

        except Exception as e:
            pred = f"<ERROR: {type(e).__name__}: {e}>"
            ok = False
            q_record = {
                "question_id": q["question_id"],
                "category": q["category"],
                "question": q["question"],
                "ground_truth": q["answer"],
                "predicted_answer": pred,
                "is_correct": False,
                "error": str(e),
            }

        results.append(q_record)
        by_cat[q["category"]].append(ok)

        status_sym = "[OK]" if ok else "[FAIL]"
        print(f"[{idx+1:02d}/{len(jobs):02d}] {status_sym} Cat{q['category']} {q['question_id']}: "
              f"Comp={q_record.get('closure_completeness', 0):.2f} | "
              f"Pred: '{pred[:40]}' | GT: '{q['answer'][:30]}'")

    total_time = time.perf_counter() - t_start
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    acc = correct / total if total > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"  Gate 1 Results: {correct}/{total} ({acc*100:.2f}%) in {total_time:.1f}s")
    for cat_id, flags in sorted(by_cat.items()):
        cat_name = CAT_NAMES.get(cat_id, str(cat_id))
        print(f"    Cat {cat_id} ({cat_name:12s}): {sum(flags):2d}/{len(flags):2d} ({sum(flags)/len(flags)*100:5.1f}%)")
    print("=" * 70)

    # Save to dedicated experiment directory
    out_dir = REPO / "benchmark_results" / "experiment_ece_gate1"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"locomo_ece_gate1_{stamp}.json"
    latest_path = out_dir / "locomo_ece_gate1_latest.json"

    summary = {
        "timestamp": stamp,
        "model": "qwen2.5:7b-instruct",
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "per_category": {str(k): {"correct": sum(v), "total": len(v)} for k, v in by_cat.items()},
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved ECE Gate 1 results -> {out_path}")

    # Compare against 76% baseline
    baseline_76_path = REPO / "benchmark_results" / "experiment_76pct_adaptive" / "locomo_full_20261011_162026.json"
    if baseline_76_path.exists():
        with open(baseline_76_path, encoding="utf-8") as f:
            b76 = json.load(f)
            b76_results = {r["question_id"]: r["is_correct"] for r in b76.get("results", [])}

        new_correct = []
        regressions = []
        for r in results:
            qid = r["question_id"]
            if qid in b76_results:
                old_ok = b76_results[qid]
                new_ok = r["is_correct"]
                if not old_ok and new_ok:
                    new_correct.append(qid)
                elif old_ok and not new_ok:
                    regressions.append(qid)

        print("\n--- Gate 1 Delta Comparison vs 76% Baseline (38/50) ---")
        print(f"  New Correct Questions (+): {len(new_correct)} -> {new_correct}")
        print(f"  Regressions (-):           {len(regressions)} -> {regressions}")
        net_delta = len(new_correct) - len(regressions)
        print(f"  Net Delta:                 {net_delta:+d} questions ({correct}/50 = {acc*100:.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/external/locomo10.json")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--num-ctx", type=int, default=8192)
    run_gate1(ap.parse_args())


if __name__ == "__main__":
    main()
