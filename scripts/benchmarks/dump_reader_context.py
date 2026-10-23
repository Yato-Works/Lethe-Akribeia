"""Dump the READER-VISIBLE context and the raw model output for selected QIDs.

Delivery is measured as gold-word coverage >= 0.5.  That is a LOOSE test: a
context can contain 60% of the gold's words as scattered fragments and still
pass.  So the coverage number alone cannot tell us what the model actually
read.  This dumps the real context string the reader receives, plus the raw
model text, so the failure can be attributed to a stage with evidence.

Usage:
    python scripts/benchmarks/dump_reader_context.py --qids lc_conv-26_23
    python scripts/benchmarks/dump_reader_context.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402


def build_context(conv: dict, q: dict, retriever, coverage, planner, compiler,
                  graph_builder) -> tuple[str, object]:
    """Replicate the production compile path and return (context, compiled)."""
    from artificial_memory_gen.recall.fact_index import FactIndex

    all_turns, _ = E.conversation_turns(conv["conversation"])

    plan = planner.plan(q["question"], [], reference_date=None)
    token_budget = plan.recommended_budget

    top_turns, turn_scores, guaranteed = coverage.retrieve_convergent_evidence(
        query=q["question"], all_turns=all_turns, retriever=retriever,
        reference_date=None, base_top_k=45)

    base_limit = 32 if plan.dominant_mode == "aggregation" else 28
    top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed
    cand = [t for t in all_turns if t[0] in top_sids]
    if not cand:
        cand = list(top_turns[:25])

    fidx = FactIndex()
    fidx.build(all_turns)
    fact_turns = fidx.query(q["question"], all_turns, max_turns=16)
    already = {(t[0], t[3]) for t in cand}
    for ft in fact_turns:
        key = (ft[0], ft[3])
        if key not in already:
            cand.append(ft)
            already.add(key)

    _probe = E._enumeration_probe_turns(
        q["question"], all_turns, retriever,
        already={(t[0], t[3]) for t in cand})
    if _probe:
        cand = cand + _probe

    _pool = sum(len(t[3].split()) * 4 // 3 for t in cand)
    if _pool:
        token_budget = E._scaled_budget(token_budget, _pool)

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
        is_multi_session=len(top_sids) > 1, adaptive_escalation=True)
    return compiled.context_text, compiled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/external/locomo10.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qids", default="")
    ap.add_argument("--ask-reader", action="store_true",
                    help="also call the reader model and show its raw output")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = E.load_locomo(Path(args.data))
    jobs = []
    for conv in data:
        for q in E.load_questions(conv):
            jobs.append((conv, q))
    if args.qids:
        want = {x.strip() for x in args.qids.split(",") if x.strip()}
        jobs = [(c, q) for (c, q) in jobs if q["question_id"] in want]
    elif args.limit:
        jobs = jobs[: args.limit]

    from eval_v4_pipeline_dev import V4CandidateRetriever
    from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
    from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
    from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
    from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder

    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    planner = CognitiveExecutionPlanner()
    compiler = BudgetedCognitiveCompiler()
    graph_builder = GraphBuilder()

    answerer = None
    if args.ask_reader:
        from artificial_memory.research.benchmarks.llm import OllamaAnswerer
        answerer = OllamaAnswerer(num_ctx=8192)

    out_rows = []
    for conv, q in jobs:
        ctx, compiled = build_context(conv, q, retriever, coverage, planner,
                                      compiler, graph_builder)
        print("=" * 78)
        print(f"{q['question_id']}  cat{q['category']}")
        print(f"Q: {q['question']}")
        print(f"GOLD: {q['answer']!r}")
        print(f"evidence ids: {q.get('evidence')}")
        print(f"ctx tokens: {getattr(compiled,'total_tokens',0)}")
        print("-" * 78)
        print("READER-VISIBLE CONTEXT:")
        print(ctx)
        print("-" * 78)
        if answerer is not None:
            prompt = E.build_reader_prompt(q["question"], ctx)
            raw = answerer.answer(q["question"], prompt).text
            print(f"RAW READER OUTPUT: {raw!r}")
            print(f"score(raw, gold) = {E.score(raw, q['answer'])}")
            print("-" * 78)
        out_rows.append({"qid": q["question_id"], "question": q["question"],
                         "gold": q["answer"], "evidence": q.get("evidence"),
                         "context": ctx,
                         "ctx_tokens": getattr(compiled, "total_tokens", 0)})

    if args.out:
        p = REPO / args.out
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
