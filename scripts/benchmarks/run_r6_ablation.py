"""
Comprehensive Ablation Study for Lethe-Akribeia R6 Milestone.
Evaluates the contribution of each core component on LongMemEval Dev Multi-Session (n=67):
  1. R6 Full (Convergent Coverage + Aggregation Certificate + 0-token Override)
  2. R6 - 0-token Override (Prompt certificate injected, but forced Reader inference)
  3. R6 - Aggregation Certificate (Aggregation co-processor completely disabled)
  4. R6 - Concept Expansion (Abstract Concept hyponym/lexical family expansion disabled)
  5. R6 - Self-Anchor Expansion (Pairwise/Self-profile dual anchor expansion disabled)
  6. R6 - Coverage Engine (No convergent retrieval, standard single-shot retrieval only)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory_gen.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.graph.dependency_graph import EvidenceBundle, EvidenceNode
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory_gen.compiler.samplers.per_session_sampler import PerSessionSampler
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter
from artificial_memory.research.benchmarks.llm import FROZEN_MODEL, OllamaAnswerer


def run_condition(
    cond_name: str,
    items: list[dict],
    enable_coverage: bool = True,
    enable_concepts: bool = True,
    enable_pairwise: bool = True,
    enable_aggregation: bool = True,
    enable_override: bool = True,
) -> dict:
    print(f"\n{'='*75}")
    print(f"  RUNNING ABLATION CONDITION: {cond_name}")
    print(f"  Coverage: {enable_coverage} (Concepts: {enable_concepts}, Pairwise: {enable_pairwise})")
    print(f"  Aggregation: {enable_aggregation} | Override: {enable_override}")
    print(f"{'='*75}")

    from scripts.benchmarks.eval_v4_pipeline_dev import V4CandidateRetriever, build_prompt

    extractor = UniversalIRExtractor()
    planner = CognitiveExecutionPlanner()
    retriever = V4CandidateRetriever()
    coverage_engine = MultiSessionCoverageEngine()
    graph_builder = GraphBuilder()
    compiler = BudgetedCognitiveCompiler()
    coprocessor = CoProcessorEngine()
    per_session_sampler = PerSessionSampler()
    answerer = OllamaAnswerer()

    t0 = time.perf_counter()
    results = []
    overridden_count = 0

    for idx, item in enumerate(items):
        qid = item["question_id"]
        q = item["question"]
        gt = item["answer"]
        qtype = item.get("question_type", "multi-session")
        q_date = item.get("question_date")

        all_records: list[StructuredIR] = []
        all_turns: list[tuple[str, str, str, str]] = []

        for s_idx, session in enumerate(item.get("haystack_sessions", [])):
            sid = item["haystack_session_ids"][s_idx]
            s_date = item["haystack_dates"][s_idx]
            for turn in session:
                speaker = turn.get("role", "user")
                content = turn.get("content", "")
                header = f"[{sid} on {s_date}]"
                line = f"{header} {speaker.capitalize()}: {content}"
                all_turns.append((sid, s_date, speaker, content))
                for r in extractor.extract(content, default_source=speaker):
                    r.raw_content = line
                    r.time_scope = s_date
                    all_records.append(r)

        plan = planner.plan(q, all_records, reference_date=q_date)
        token_budget = plan.recommended_budget
        is_aggregation = bool(plan.dominant_mode == "aggregation")
        guaranteed_sids: set[str] = set()

        if enable_coverage:
            top_turns, turn_scores, guaranteed_sids = coverage_engine.retrieve_convergent_evidence(
                query=q,
                all_turns=all_turns,
                retriever=retriever,
                reference_date=q_date,
                base_top_k=60 if is_aggregation else 45,
                enable_concepts=enable_concepts,
                enable_pairwise=enable_pairwise,
            )
            base_limit = 32 if is_aggregation else 28
            top_sids = {t[0] for t in top_turns[:base_limit]} | guaranteed_sids
            candidate_turns = [t for t in all_turns if t[0] in top_sids]
            if not candidate_turns:
                candidate_turns = list(top_turns[:25])
            if is_aggregation or len(guaranteed_sids) > 1 or plan.requires_event_aggregation or plan.aggregation_demand >= 0.50 or plan.temporal_demand >= 0.85:
                token_budget = min(2200, int(token_budget * 1.5))
        elif is_aggregation:
            top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=60, reference_date=q_date)
            top_sids = {t[0] for t in top_turns[:30]}
            candidate_turns = [t for t in all_turns if t[0] in top_sids]
            if not candidate_turns:
                candidate_turns = list(all_turns)
            token_budget = min(2200, int(token_budget * 1.5))
        else:
            top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=45, reference_date=q_date)
            top_sids = {t[0] for t in top_turns[:26]}
            candidate_turns = [t for t in all_turns if t[0] in top_sids]
            if not candidate_turns:
                candidate_turns = top_turns[:20]

        cand_sids = {c[0] for c in candidate_turns}
        header_sid_re = re.compile(r"^\[([^\s\]]+)")
        cand_records = [r for r in all_records if (m := header_sid_re.match(r.raw_content or "")) and m.group(1) in cand_sids]
        if not cand_records:
            cand_records = all_records[:50]

        t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(candidate_turns)}
        anchor_ids = []
        anchor_scores: dict[str, float] = {}
        for t in top_turns:
            nid = t_map.get((t[0], t[3]))
            if nid and nid not in anchor_ids:
                anchor_ids.append(nid)
                score = turn_scores.get((t[0], t[3]), 1.0)
                if t[0] in guaranteed_sids:
                    score = max(score, 25.0)
                anchor_scores[nid] = score

        # Co-processor Reasoning
        if enable_aggregation:
            bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)
        else:
            # Temporary override aggregation engine
            orig_engine = coprocessor.aggregation_engine
            from artificial_memory_gen.compiler.certificates.aggregation_certificate import AggregationCertificateEngine
            class DisabledAggEngine(AggregationCertificateEngine):
                def generate(self, *args, **kwargs):
                    from artificial_memory_gen.compiler.certificates.aggregation_certificate import AggregationCertificate
                    return AggregationCertificate(has_certificate=False, query_type="none", matched_events=[], total_count=None, difference_value=None, formatted_certificate="", direct_answer_candidate=None)
            coprocessor.aggregation_engine = DisabledAggEngine()
            bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)
            coprocessor.aggregation_engine = orig_engine

        if not enable_override and bundle.deterministic_answer_override:
            from dataclasses import replace
            bundle = replace(bundle, deterministic_answer_override=None)

        if bundle.deterministic_answer_override:
            pred = bundle.deterministic_answer_override
            tok_used = 0
            lat_ms = 0.0
            overridden_count += 1
            compiled_ctx_str = "[0-TOKEN DETERMINISTIC RESOLUTION]"
        elif is_aggregation:
            sampled_ev = per_session_sampler.sample(q, candidate_turns, max_total_tokens=token_budget)
            compiled_ctx_str = per_session_sampler.format_as_context(sampled_ev, provenance=False)
            if bundle.assembled_certificate_block:
                compiled_ctx_str = f"{bundle.assembled_certificate_block}\n\n{compiled_ctx_str}"
        else:
            edg = graph_builder.build(candidate_turns, cand_records)
            bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
            clamped_bundles = []
            for b in bundles:
                new_members = []
                for m in b.member_nodes:
                    words = m.raw_text.split()
                    if len(words) > 180:
                        trunc_text = " ".join(words[:180]) + " ..."
                        new_members.append(EvidenceNode(
                            node_id=m.node_id, session_id=m.session_id, session_date=m.session_date,
                            speaker=m.speaker, raw_text=trunc_text, tokens=len(trunc_text.split()) * 4 // 3,
                            entities=m.entities, properties=m.properties, values=m.values
                        ))
                    else:
                        new_members.append(m)
                tot_tok = sum(n.tokens for n in new_members)
                clamped_bundles.append(EvidenceBundle(b.bundle_id, b.primary_node_id, new_members, tot_tok, b.rationale))
            bundles = clamped_bundles

            is_multi = bool(
                plan.requires_event_aggregation or plan.aggregation_demand >= 0.50
                or plan.temporal_demand >= 0.85 or len(guaranteed_sids) > 1
            )
            compiled = compiler.compile(
                question=q,
                bundles=bundles,
                token_budget=token_budget,
                co_processor_bundle=bundle,
                anchor_scores=anchor_scores,
                is_multi_session=is_multi,
                adaptive_escalation=True,
            )
            compiled_ctx_str = compiled.context_text

        if not bundle.deterministic_answer_override:
            prompt_context = build_prompt(
                cond="R6",
                plan=plan,
                is_aggregation=is_aggregation,
                compiled_ctx_str=compiled_ctx_str,
                q=q,
                q_date=q_date,
            )
            ans_resp = answerer.answer(q, prompt_context)
            pred = ans_resp.text
            tok_used = ans_resp.prompt_tokens
            lat_ms = ans_resp.latency_ms

        is_abstention = qtype == "abstention" or qid.endswith("_abs") or "information provided is not enough" in str(gt).lower()
        is_ok = LongMemEvalAdapter.score_answer(
            question_type=qtype, gt=gt, predicted_answer=pred,
            is_abstention_gt=is_abstention, pcc_is_abstention=is_abstention
        )

        results.append({
            "question_id": qid,
            "ground_truth": gt,
            "predicted_answer": pred,
            "is_correct": is_ok,
            "prompt_tokens": tok_used,
        })
        status = "PASS" if is_ok else "FAIL"
        print(f"[{idx+1:02d}/67] {qid}: {status} | tok={tok_used:4d} | pred='{pred[:30]}'")

    elapsed = time.perf_counter() - t0
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    acc = correct / total if total > 0 else 0.0
    mean_tok = sum(r["prompt_tokens"] for r in results) / total if total > 0 else 0.0

    print(f"\nCondition {cond_name} Complete: {correct}/{total} ({acc:.2%}), Overrides: {overridden_count}, Mean Tokens: {mean_tok:.1f}")
    return {
        "condition": cond_name,
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "overrides": overridden_count,
        "mean_tokens": mean_tok,
        "elapsed_seconds": elapsed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", type=str, default="all", help="Condition to run: all, full, no_override, no_aggregation, no_concept, no_pairwise, no_coverage")
    args = parser.parse_args()

    data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
    qids = json.loads((REPO / "benchmark_results" / "splits" / "lme_dev_ms_qids.json").read_text(encoding="utf-8"))
    ms_items = [item for item in data if item["question_id"] in qids]

    conditions = [
        ("R6 Full", True, True, True, True, True),
        ("R6 - 0-token Override", True, True, True, True, False),
        ("R6 - Aggregation Cert", True, True, True, False, False),
        ("R6 - Concept Expansion", True, False, True, True, True),
        ("R6 - Self-Anchor Dual", True, True, False, True, True),
        ("R6 - Coverage Engine", False, False, False, True, True),
    ]

    if args.cond != "all":
        conditions = [c for c in conditions if args.cond.lower() in c[0].lower()]

    out_dir = REPO / "benchmark_results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_table = []
    for name, cov, conc, pair, agg, over in conditions:
        out_f = out_dir / f"ablation_{name.replace(' ', '_').lower()}.json"
        if out_f.exists():
            print(f"\nLoading existing result for condition: {name}")
            with open(out_f, "r", encoding="utf-8") as f:
                res = json.load(f)
        else:
            res = run_condition(name, ms_items, enable_coverage=cov, enable_concepts=conc, enable_pairwise=pair, enable_aggregation=agg, enable_override=over)
            with open(out_f, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
        summary_table.append(res)

    print("\n" + "="*80)
    print("  R6 ABLATION STUDY SUMMARY (LongMemEval Dev Multi-Session n=67)")
    print("="*80)
    print(f"{'Condition':<26} {'Accuracy':<10} {'Correct/67':<12} {'Diff vs Full':<14} {'Mean Tokens':<12}")
    print("-" * 75)
    full_correct = summary_table[0]["correct"]
    for row in summary_table:
        diff = row["correct"] - full_correct
        diff_str = f"{diff:+d} ({diff/67*100:+.1f}%)" if diff != 0 else "baseline"
        print(f"{row['condition']:<26} {row['accuracy']:<10.1%} {row['correct']}/67       {diff_str:<14} {row['mean_tokens']:<12.1f}")
    print("="*80)

    with open(out_dir / "r6_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
