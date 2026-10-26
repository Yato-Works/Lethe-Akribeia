"""Measure which pipeline STAGE fails for LoCoMo questions.

Measures the stages SEPARATELY, because "the rank didn't change" is not evidence
that "the downstream didn't change", and "the evidence is in the context" is not
evidence that "the answer is right".

Stages:
  1. RETRIEVAL  - gold-evidence turn appears in the retrieved candidate set
  2. DELIVERY   - gold text reaches the compiled reader context
  3. READER     - the answer is correct

This replicates the PRODUCTION path from eval_locomo_full.run() verbatim
(coverage.retrieve_convergent_evidence -> fact index -> enumeration probe ->
_scaled_budget -> graph_builder.build -> form_bundles -> compiler.compile),
so a failure measured here is the product's failure, not a copy's.

Usage:
    python scripts/benchmarks/stage_diagnosis.py --limit 30
    python scripts/benchmarks/stage_diagnosis.py --qids lc_conv-26_23,lc_conv-26_51
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402

COVERAGE_THRESHOLD = 0.5  # >=50% of gold's significant words present


def norm_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", str(text).lower()) if len(w) > 2}


def word_coverage(context: str, gold: str) -> float:
    """Fraction of the gold's significant words present in the context."""
    gw = norm_words(gold)
    if not gw:
        return 0.0
    return len(gw & norm_words(context)) / len(gw)


def dia_text(conv: dict, dia_id: str) -> str:
    """Resolve a LoCoMo dialogue id ('D7:8') to its text."""
    inner = conv.get("conversation", {})
    if not isinstance(inner, dict):
        return ""
    for key, val in inner.items():
        if not isinstance(key, str) or not key.startswith("session_"):
            continue
        if not isinstance(val, list):
            continue
        for turn in val:
            if isinstance(turn, dict) and turn.get("dia_id") == dia_id:
                return turn.get("text", "")
    return ""


def run(args) -> None:
    data = E.load_locomo(Path(args.data))

    jobs: list[tuple[dict, dict]] = []
    for conv in data:
        for q in E.load_questions(conv):
            jobs.append((conv, q))
    if args.qids:
        want = {x.strip() for x in args.qids.split(",") if x.strip()}
        jobs = [(c, q) for (c, q) in jobs if q["question_id"] in want]
    elif args.limit:
        jobs = jobs[: args.limit]

    print(f"stage diagnosis: {len(jobs)} questions")

    from eval_v4_pipeline_dev import V4CandidateRetriever
    from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
    from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
    from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
    from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
    from artificial_memory_gen.recall.fact_index import FactIndex
    from artificial_memory.skills.answer_committer import (
        commit_answer, post_process_answer,
    )

    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    planner = CognitiveExecutionPlanner()
    compiler = BudgetedCognitiveCompiler()
    graph_builder = GraphBuilder()

    turn_cache: dict[str, list] = {}
    fact_cache: dict[str, FactIndex] = {}

    per_cat: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "meas": 0, "retr": 0, "deliv": 0, "ok": 0})
    rows: list[dict] = []

    for conv, q in jobs:
        sid = conv.get("sample_id", "x")
        cat = q["category"]
        per_cat[cat]["n"] += 1

        ev_ids = [e for e in (q.get("evidence") or []) if isinstance(e, str)]
        ev_text = "\n".join(dia_text(conv, e) for e in ev_ids).strip()
        measurable = bool(ev_text) and q["category"] != 5  # cat5 has no citable gold
        if measurable:
            per_cat[cat]["meas"] += 1

        if sid not in turn_cache:
            turn_cache[sid] = E.conversation_turns(conv["conversation"])
        all_turns, _ = turn_cache[sid]

        try:
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

            if sid not in fact_cache:
                _f = FactIndex()
                _f.build(all_turns)
                fact_cache[sid] = _f
            fact_turns = fact_cache[sid].query(q["question"], all_turns, max_turns=16)
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
            ctx = compiled.context_text

            # ---- STAGE 1: retrieval ----
            retr_hit = False
            if measurable:
                for e in ev_ids:
                    txt = dia_text(conv, e)
                    if not txt:
                        continue
                    for t in top_turns:
                        if word_coverage(t[3], txt) >= COVERAGE_THRESHOLD:
                            retr_hit = True
                            break
                    if retr_hit:
                        break

            # ---- STAGE 2: delivery ----
            cov = word_coverage(ctx, ev_text) if measurable else None
            deliv_hit = bool(measurable and cov is not None
                             and cov >= COVERAGE_THRESHOLD)

            # ---- STAGE 3: reader ----
            committed = None
            try:
                _c = commit_answer(q["question"], ctx, q["category"])
                if _c.used and str(_c.answer).strip():
                    committed = post_process_answer(
                        q["question"], str(_c.answer).strip(), q["category"])
            except Exception:
                committed = None
            pred = committed or ""
            ok = E.score(pred, q["answer"])

            if measurable and retr_hit:
                per_cat[cat]["retr"] += 1
            if deliv_hit:
                per_cat[cat]["deliv"] += 1
            if ok:
                per_cat[cat]["ok"] += 1

            rows.append({
                "qid": q["question_id"], "cat": cat,
                "question": q["question"], "gold": q["answer"],
                "evidence_ids": ev_ids,
                "measurable": measurable,
                "retrieved": retr_hit, "coverage": cov, "delivered": deliv_hit,
                "pred": pred, "correct": ok,
                "ctx_tokens": getattr(compiled, "total_tokens", 0),
            })
        except Exception as e:
            rows.append({"qid": q["question_id"], "cat": cat,
                         "error": f"{type(e).__name__}: {e}"})

    print()
    print("=" * 78)
    print("STAGE FUNNEL  (questions with citable gold evidence)")
    print("=" * 78)
    print(f"{'cat':4s} {'n':>4s} {'meas':>5s} {'retrieved':>10s} {'delivered':>10s} {'correct':>8s}")
    print("-" * 78)
    for cat in sorted(per_cat):
        v = per_cat[cat]
        print(f"cat{cat:<2d} {v['n']:4d} {v['meas']:5d} {v['retr']:10d} "
              f"{v['deliv']:10d} {v['ok']:8d}")

    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"\nerrors: {len(errs)}")
        for r in errs[:10]:
            print(f"  {r['qid']}: {r['error']}")

    if args.out:
        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "config": {"limit": args.limit, "qids": args.qids,
                           "coverage_threshold": COVERAGE_THRESHOLD,
                           "data": args.data},
                "per_category": {str(k): dict(v) for k, v in per_cat.items()},
                "rows": rows,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/external/locomo10.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qids", default="")
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
