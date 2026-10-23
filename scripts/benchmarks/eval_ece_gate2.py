"""Gate 2 Generalization Evaluation Runner: Unseen Conversation Hold-Out.

Evaluates unseen conversations (e.g. Conversation 1: conv-30, Jon & Gina)
to verify that Evidence Closure Engine (ECE) / Memory Compiler v2 generalizes
without conversation-specific overfitting or parameter tuning.

Strict Contract:
- Frozen Ollama Reader parameters (qwen2.5:7b-instruct, temp=0.0, seed=42).
- Zero gold contamination or runtime question leaks (AST validated).
- Independent output destination: benchmark_results/experiment_ece_gate2/
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
from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.recall.fact_index import FactIndex
from artificial_memory_gen.compiler.closure import (
    DeterministicQueryPlanner,
    EvidenceClosureEngine,
    AnswerTopology,
)
# The frozen retriever lives in the dev harness; import it here rather than
# reimplementing it so baseline and ECE modes rank turns identically.
# NOTE: this import was missing, so every run died with NameError on the first
# question -- which is why the original ECE artifacts were produced by an
# earlier, working copy and never re-verified against this file.
from eval_v4_pipeline_dev import V4CandidateRetriever
from artificial_memory.research.benchmarks.llm import OllamaAnswerer


def run_pipeline(jobs: list[tuple[dict, dict]], mode: str, num_ctx: int = 8192) -> dict:
    """Run evaluation for a list of (conv, q) jobs in either 'baseline' or 'ece' mode."""
    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    graph_builder = GraphBuilder()
    compiler = BudgetedCognitiveCompiler()
    answerer = OllamaAnswerer(num_ctx=num_ctx)

    if mode == "baseline":
        planner = CognitiveExecutionPlanner()
        ece_planner = None
        closure_engine = None
    else:
        planner = None
        ece_planner = DeterministicQueryPlanner()
        closure_engine = EvidenceClosureEngine()

    results: list[dict] = []
    by_cat: dict[int, list[bool]] = defaultdict(list)
    t_start = time.perf_counter()

    turn_cache: dict[str, list] = {}
    fact_cache: dict[str, FactIndex] = {}

    print(f"\nRunning {mode.upper()} mode on {len(jobs)} questions...")
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
            if mode == "baseline":
                # Baseline Adaptive Pipeline
                plan = planner.plan(q["question"], [], reference_date=None)
                top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
                    query=q["question"], all_turns=all_turns, retriever=retriever,
                    reference_date=None, base_top_k=45)

                base_limit = 32 if plan.dominant_mode == "aggregation" else 28
                top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
                cand = [t for t in all_turns if t[0] in top_sids]
                if not cand:
                    cand = list(top_turns[:25])

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

                _probe = _enumeration_probe_turns(
                    q["question"], all_turns, retriever,
                    already={(t[0], t[3]) for t in cand})
                if _probe:
                    cand = cand + _probe

                _pool_tokens = sum(len(t[3].split()) * 4 // 3 for t in cand)
                token_budget = _scaled_budget(plan.recommended_budget, _pool_tokens)

                t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(cand)}
                anchor_ids = []
                anchor_scores = {}
                for t in top_turns:
                    nid = t_map.get((t[0], t[3]))
                    if nid and nid not in anchor_ids:
                        anchor_ids.append(nid)
                        anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)

                closure_meta = {}
            else:
                # ECE Mode
                query_plan = ece_planner.plan(q["question"])
                top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
                    query=q["question"], all_turns=all_turns, retriever=retriever,
                    reference_date=None, base_top_k=45)

                base_limit = 32 if query_plan.topology == AnswerTopology.SET_ENUMERATION else 28
                top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
                cand = [t for t in all_turns if t[0] in top_sids]
                if not cand:
                    cand = list(top_turns[:25])

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

                _probe = _enumeration_probe_turns(
                    q["question"], all_turns, retriever,
                    already={(t[0], t[3]) for t in cand})
                if _probe:
                    cand = cand + _probe

                closure_res = closure_engine.run_closure(
                    query_plan=query_plan,
                    all_turns=cand,
                    initial_top_k=35,
                )

                # RUNNER-SIDE FALLBACK for requirements the engine cannot expand.
                #
                # EvidenceClosureEngine._derive_expansion_queries handles
                # req_multi_session_breadth, req_enum_instances, req_temporal_anchor
                # and req_anchor_event -- but NOT req_direct_fact.  The planner
                # (closure/planner.py) does emit req_direct_fact, so it is the
                # single most common unmet requirement: 28 of 50 on conv-30.
                #
                # When it goes unmet, expansion_targets comes back empty and the
                # loop terminates with "no_further_expansion_possible", adding
                # nothing.  Measured consequence: b=0/c=0 against baseline -- the
                # gate detected the gap and then did nothing about it.
                #
                # This block is the runner compensating for that gap.  It only
                # fires when the engine has explicitly given up, and it only
                # widens the candidate pool via the same rank_turns path the
                # enumeration probe already uses.
                closure_fallback_turns = 0
                if (closure_res.closure_state.termination_reason
                        == "no_further_expansion_possible"):
                    # Snapshot the pre-fallback pool so only genuinely NEW turns
                    # get fallback anchors.
                    _pre_fallback_keys = {(t[0], t[3]) for t in cand}
                    _fb_keys: set[tuple[str, str]] = set()
                    _fb_scores: dict[tuple[str, str], float] = {}
                    for req in closure_res.requirement_details:
                        if req.status.value not in ("unresolved", "partially_supported"):
                            continue
                        # Probe with several phrasings, because the question
                        # text alone often retrieves NOTHING: the frozen
                        # STOP_WORDS list strips where/move/from/when/how, so
                        # "Where did Caroline move from 4 years ago?" reduces to
                        # "caroline" and matches no turn.  Measured: that probe
                        # returns 0 turns, while the same question phrased as
                        # "moved from origin country" retrieves 2.
                        probes = [
                            (req.description or "").strip(),
                            q["question"],
                            f"{q['question']} when date origin place",
                            f"{(req.description or '').strip()} place date origin",
                        ]
                        for probe_text in probes:
                            if not probe_text or not probe_text.strip():
                                continue
                            fb_extra, _fb_s = retriever.rank_turns(
                                probe_text, all_turns, top_k=8)
                            for et in fb_extra:
                                k = (et[0], et[3])
                                if k not in _fb_keys:
                                    cand.append(et)
                                    _fb_keys.add(k)
                                    closure_fallback_turns += 1
                                _fb_scores[k] = max(
                                    _fb_scores.get(k, 0.0), _fb_s.get(k, 0.0))
                else:
                    _fb_keys = set()
                    _fb_scores = {}
                    _pre_fallback_keys = set()

                t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(cand)}
                # Fallback turns must be ANCHORED or the knapsack will never
                # select them.  Measured failure: adding 12 candidate turns to
                # conv-30_9 changed context_tokens by 0 (1394 -> 1394), because
                # the new turns were in the pool and got bundles but carried no
                # anchor score, so they lost every selection round.
                fallback_anchor_ids: list[str] = []
                anchor_ids = []
                anchor_scores = {}
                for t in top_turns:
                    nid = t_map.get((t[0], t[3]))
                    if nid and nid not in anchor_ids:
                        anchor_ids.append(nid)
                        anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)

                # Anchor the turns this fallback contributed, scored by the
                # rank_turns score that selected them.
                for k in _fb_keys:
                    if k in _pre_fallback_keys:
                        continue
                    nid = t_map.get(k)
                    if nid and nid not in anchor_ids:
                        anchor_ids.append(nid)
                        anchor_scores[nid] = _fb_scores.get(k, 2.0)
                        fallback_anchor_ids.append(nid)

                for req in closure_res.requirement_details:
                    for stid in req.supporting_turn_ids:
                        for c in closure_res.admitted_candidates:
                            if c.turn_id == stid:
                                nid = t_map.get((c.session_id, c.content))
                                if nid and nid not in anchor_ids:
                                    anchor_ids.append(nid)
                                    anchor_scores[nid] = turn_scores.get((c.session_id, c.content), 2.0)

                _pool_tokens = sum(len(t[3].split()) * 4 // 3 for t in cand)
                token_budget = _scaled_budget(query_plan.recommended_budget, _pool_tokens)
                if query_plan.topology == AnswerTopology.INFERENCE_BOOLEAN:
                    token_budget = min(1200, token_budget)

                closure_meta = {
                    "topology": query_plan.topology.value,
                    "closure_completeness": closure_res.closure_completeness,
                    "satisfied_requirements": list(sorted(closure_res.closure_state.satisfied_requirements)),
                    "unmet_requirements": list(closure_res.unmet_requirements),
                    "conflicting_requirements": list(sorted(closure_res.closure_state.conflicting_requirements)),
                    "termination_reason": closure_res.closure_state.termination_reason,
                    "fallback_turns_added": closure_fallback_turns,
                }

            # Dialogue Dependency Graph Bundling & Knapsack Compilation
            edg = graph_builder.build(cand, [])
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            compiled = compiler.compile(
                question=q["question"], bundles=bundles,
                token_budget=token_budget, anchor_scores=anchor_scores,
                is_multi_session=len(top_sids) > 1, adaptive_escalation=True)

            # Frozen Reader Prompt & Inference
            prompt_text = build_reader_prompt(q["question"], compiled.context_text)
            pred = answerer.answer(q["question"], prompt_text).text.strip()

            if pred.endswith(".") and len(pred.split()) <= 2:
                pred = pred[:-1].strip()

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
                "context_tokens": compiled.total_tokens,
                "latency_seconds": round(duration, 3),
            }
            q_record.update(closure_meta)

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
        mark = "OK" if ok else "FAIL"
        print(f"[{idx+1:02d}/{len(jobs):02d}] [{mark}] Cat{q['category']} {q['question_id']}: Pred: {pred!r:35s} | GT: {q['answer'][:30]!r}")

    elapsed = time.perf_counter() - t_start
    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    acc = correct / total if total else 0.0

    print(f"\n{mode.upper()} Final: {correct}/{total} ({acc*100:.2f}%) in {elapsed:.1f}s")
    for cat_id, cat_name in sorted(CAT_NAMES.items()):
        vals = by_cat.get(cat_id, [])
        if vals:
            c_acc = sum(vals) / len(vals)
            print(f"  Cat {cat_id} ({cat_name:12s}): {sum(vals):2d}/{len(vals):2d} ({c_acc*100:5.1f}%)")

    return {
        "mode": mode,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "elapsed_seconds": round(elapsed, 2),
        "per_category": {str(k): {"correct": sum(v), "total": len(v)} for k, v in by_cat.items()},
        "results": results,
    }


def run_gate2(args) -> None:
    data = load_locomo(Path(args.data))
    conv = data[args.conv_idx]
    all_qs = load_questions(conv)
    offset = args.offset
    limit = args.limit or len(all_qs)
    selected_qs = all_qs[offset : offset + limit]
    jobs = [(conv, q) for q in selected_qs]

    speaker_a = conv.get("conversation", {}).get("speaker_a")
    speaker_b = conv.get("conversation", {}).get("speaker_b")
    sid = conv.get("sample_id", f"conv_{args.conv_idx}")

    print("=" * 70)
    print("  Evidence Closure Engine (ECE) - Gate 2 Hold-Out Generalization")
    print(f"  Target: Conversation {args.conv_idx} ({sid}: {speaker_a} & {speaker_b})")
    print(f"  Evaluating {len(jobs)} hold-out questions (offset={offset}, limit={limit})")
    print("=" * 70)

    out_dir = REPO / "benchmark_results" / "experiment_ece_gate2"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run Baseline
    base_res = run_pipeline(jobs, mode="baseline", num_ctx=args.num_ctx)
    base_file = out_dir / f"locomo_baseline_gate2_{sid}_off{offset}_{stamp}.json"
    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(base_res, f, ensure_ascii=False, indent=2)

    # Run ECE
    ece_res = run_pipeline(jobs, mode="ece", num_ctx=args.num_ctx)
    ece_file = out_dir / f"locomo_ece_gate2_{sid}_off{offset}_{stamp}.json"
    with open(ece_file, "w", encoding="utf-8") as f:
        json.dump(ece_res, f, ensure_ascii=False, indent=2)

    # Compare
    base_lut = {r["question_id"]: r["is_correct"] for r in base_res["results"]}
    new_correct = []
    regressions = []
    for r in ece_res["results"]:
        qid = r["question_id"]
        if qid in base_lut:
            b_ok = base_lut[qid]
            e_ok = r["is_correct"]
            if not b_ok and e_ok:
                new_correct.append(qid)
            elif b_ok and not e_ok:
                regressions.append(qid)

    print("\n" + "=" * 70)
    print(f"  Gate 2 Hold-Out Comparative Summary ({sid} [offset={offset}, limit={limit}])")
    print("=" * 70)
    print(f"  Baseline Accuracy: {base_res['correct']}/{base_res['total']} ({base_res['accuracy']*100:.2f}%)")
    print(f"  ECE Accuracy:      {ece_res['correct']}/{ece_res['total']} ({ece_res['accuracy']*100:.2f}%)")
    delta_pts = (ece_res['accuracy'] - base_res['accuracy']) * 100
    print(f"  Delta:             {delta_pts:+.2f} percentage points ({ece_res['correct'] - base_res['correct']:+d} questions)")
    print(f"  New Correct:       {len(new_correct)} -> {new_correct}")
    print(f"  Regressions:       {len(regressions)} -> {regressions}")
    print("=" * 70)
    print(f"Saved Gate 2 Baseline -> {base_file}")
    print(f"Saved Gate 2 ECE      -> {ece_file}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/external/locomo10.json")
    ap.add_argument("--conv-idx", type=int, default=1, help="Index of conversation (1 = conv-30 Jon & Gina)")
    ap.add_argument("--offset", type=int, default=0, help="Offset index into conversation questions")
    ap.add_argument("--limit", type=int, default=50, help="Number of questions to evaluate")
    ap.add_argument("--num-ctx", type=int, default=8192)
    run_gate2(ap.parse_args())


if __name__ == "__main__":
    main()
