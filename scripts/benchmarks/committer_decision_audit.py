"""Measure the COMMITTER-vs-READER decision rule, not the reader.

The committer is a deterministic fallback wired to fire only when the reader
abstains or errors.  On cat2 it measured 77.4% against the reader's 44.6%, so
the *ordering* of the decision is a real lever that costs no model changes.

This runs BOTH paths on EVERY case and records each answer, so several
decision rules can be scored offline from one run:

  A. reader-first        (current production wiring)
  B. committer-first     (use the committer whenever it claims)
  C. committer-on-abstain (committer only where the reader abstained)
  D. reader, then committer on abstain (== A, sanity check)

It also records, for every abstention, whether the committer HAD an answer --
the "recoverable abstention" count.

Usage:
    python scripts/benchmarks/committer_decision_audit.py --manifest results/m30_60/development_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "benchmarks"))

import eval_locomo_full as E  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="results/m30_60/committer_decision_audit.json")
    args = ap.parse_args()

    with open(REPO / args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    data = E.load_locomo(REPO / "datasets/external/locomo10.json")
    by_idx = {}
    for ci, conv in enumerate(data):
        for qi, q in enumerate(E.load_questions(conv)):
            by_idx[(ci, qi)] = (conv, q)
    jobs = [by_idx[(int(e["conv_idx"]), int(e["q_idx"]))] for e in manifest]

    from eval_v4_pipeline_dev import V4CandidateRetriever
    from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
    from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
    from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
    from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
    from artificial_memory_gen.recall.fact_index import FactIndex
    from artificial_memory.research.benchmarks.llm import OllamaAnswerer
    from artificial_memory.skills.answer_committer import (
        commit_answer, post_process_answer,
    )

    retriever = V4CandidateRetriever()
    coverage = MultiSessionCoverageEngine()
    planner = CognitiveExecutionPlanner()
    compiler = BudgetedCognitiveCompiler()
    graph_builder = GraphBuilder()
    answerer = OllamaAnswerer(num_ctx=8192)

    turn_cache: dict[str, list] = {}
    fact_cache: dict[str, FactIndex] = {}

    rows = []
    for n, (conv, q) in enumerate(jobs, 1):
        sid = conv.get("sample_id", "x")
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

            # ---- READER (always run, so both paths exist per case) ----
            reader_raw = answerer.answer(
                q["question"], E.build_reader_prompt(q["question"], ctx)).text.strip()
            if reader_raw.endswith(".") and len(reader_raw.split()) <= 2:
                reader_raw = reader_raw[:-1].strip()
            if __import__("re").search(r"\bwhen\b", q["question"], 2):
                reader_raw = E.resolve_relative_dates(reader_raw, ctx, q["question"])
            reader_abstained = reader_raw.strip().lower() in E.ABSTENTIONS
            reader_ok = E.score(reader_raw, q["answer"])

            # ---- COMMITTER ----
            committed = None
            comm_used = False
            try:
                _c = commit_answer(q["question"], ctx, q["category"])
                comm_used = bool(_c.used)
                if _c.used and str(_c.answer).strip():
                    committed = post_process_answer(
                        q["question"], str(_c.answer).strip(), q["category"])
            except Exception:
                comm_used = False
            comm_ok = E.score(committed, q["answer"]) if committed else False

            rows.append({
                "qid": q["question_id"], "category": q["category"],
                "question": q["question"], "gold": q["answer"],
                "reader_raw": reader_raw, "reader_abstained": reader_abstained,
                "reader_correct": reader_ok,
                "committer_used": comm_used, "committer_answer": committed,
                "committer_correct": comm_ok,
                "ctx_tokens": getattr(compiled, "total_tokens", 0),
            })
        except Exception as e:
            rows.append({"qid": q["question_id"], "category": q["category"],
                         "error": f"{type(e).__name__}: {e}"})

        if n % 50 == 0:
            print(f"  [{n}/{len(jobs)}]")

    # ---- DECISION RULES ----
    def acc(pred_fn):
        ok = tot = 0
        for r in rows:
            if "error" in r:
                continue
            tot += 1
            ok += int(pred_fn(r))
        return ok, tot

    rules = {
        "A_reader_first": lambda r: r["reader_correct"],
        "B_committer_first": lambda r: (r["committer_correct"] if r["committer_used"]
                                        else r["reader_correct"]),
        "C_committer_on_abstain": lambda r: (
            r["committer_correct"] if r["reader_abstained"] and r["committer_used"]
            else r["reader_correct"]),
        "D_committer_first_only_when_better": lambda r: (
            r["committer_correct"] if r["committer_used"] and not r["reader_correct"]
            else r["reader_correct"]),
    }

    print()
    print("=" * 74)
    print("DECISION-RULE COMPARISON (same run, both paths recorded per case)")
    print("=" * 74)
    base_ok = base_tot = None
    for name, fn in rules.items():
        ok, tot = acc(fn)
        if base_ok is None:
            base_ok, base_tot = ok, tot
        print(f"  {name:34s} {ok:4d}/{tot:<4d} = {ok/tot*100:5.2f}%"
              f"   delta vs A: {(ok-base_ok)/tot*100:+6.2f}pt")

    # ---- RECOVERABLE ABSTENTIONS ----
    ab = [r for r in rows if r.get("reader_abstained")]
    ab_with_commit = [r for r in ab if r.get("committer_used")]
    ab_recoverable = [r for r in ab_with_commit if r.get("committer_correct")]
    print()
    print("=" * 74)
    print("RECOVERABLE ABSTENTIONS")
    print("=" * 74)
    print(f"  reader abstained           : {len(ab)}")
    print(f"  ...committer had an answer : {len(ab_with_commit)}")
    print(f"  ...that answer was correct : {len(ab_recoverable)}")

    # by category
    print()
    print(f"  {'cat':5s} {'abstain':>8s} {'commit':>7s} {'rescued':>8s}")
    for cat in sorted({r["category"] for r in rows if "error" not in r}):
        a = [r for r in ab if r["category"] == cat]
        c = [r for r in ab_with_commit if r["category"] == cat]
        rc = [r for r in ab_recoverable if r["category"] == cat]
        print(f"  cat{cat:<2d} {len(a):8d} {len(c):7d} {len(rc):8d}")

    # disagreements: reader right, committer wrong
    disagree_bad = [r for r in rows
                    if not r.get("error") and r.get("committer_used")
                    and not r["reader_correct"] and r["committer_correct"]
                    and not r["reader_correct"]]
    print(f"\n  committer uniquely correct (reader wrong): "
          f"{sum(1 for r in rows if not r.get('error') and r.get('committer_used') and r['committer_correct'] and not r['reader_correct'])}")
    print(f"  reader uniquely correct (committer wrong): "
          f"{sum(1 for r in rows if not r.get('error') and r.get('committer_used') and r['reader_correct'] and not r['committer_correct'])}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "manifest": args.manifest,
            "n": len(rows),
            "decision_rules": {k: acc(v) for k, v in rules.items()},
            "recoverable_abstentions": {
                "reader_abstained": len(ab),
                "committer_had_answer": len(ab_with_commit),
                "rescued": len(ab_recoverable),
            },
            "rows": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
