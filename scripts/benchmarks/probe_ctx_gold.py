"""Probe context compilation presence for gold turns in multi-session failures."""

import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.compiler.graph.graph_builder import GraphBuilder
from artificial_memory.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory.compiler.samplers.per_session_sampler import PerSessionSampler

import importlib.util
spec = importlib.util.spec_from_file_location("eval_v4", str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"))
eval_v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_v4)
retriever = eval_v4.V4CandidateRetriever()

coverage_engine = MultiSessionCoverageEngine()
extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()
graph_builder = GraphBuilder()
compiler = BudgetedCognitiveCompiler()
coprocessor = CoProcessorEngine()
sampler = PerSessionSampler(turns_per_session=3)

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = {it["question_id"]: it for it in json.load(f)}

r6_path = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_R6.json"
r6_data = json.loads(r6_path.read_text(encoding="utf-8"))
fails = [r for r in r6_data["results"] if r["question_type"] == "multi-session" and not r["is_correct"]]

header_sid_re = re.compile(r"^\[([^\s\]]+)")

print(f"Checking compiled context coverage for {len(fails)} failures...")

for idx, r in enumerate(fails):
    qid = r["question_id"]
    item = all_items[qid]
    q = item["question"]
    q_date = item.get("question_date", "")
    gold_sids = set(item.get("answer_session_ids", []))
    gt = str(r.get("ground_truth", ""))

    all_turns = []
    all_records = []
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        for turn in session:
            speaker = turn.get("role", "user")
            content = turn.get("content", "")
            header = f"[{sid} on {s_date}]" if s_date else f"[{sid}]"
            line = f"{header} {speaker.capitalize()}: {content}"
            all_turns.append((sid, s_date, speaker, content))
            recs = extractor.extract(content, default_source=speaker)
            for rec in recs:
                rec.raw_content = line
                rec.time_scope = s_date
                all_records.append(rec)

    plan = planner.plan(q, all_records, reference_date=q_date)
    token_budget = plan.recommended_budget
    is_agg = bool(plan.dominant_mode == "aggregation")

    cov_turns, cov_scores, guaranteed_sids = coverage_engine.retrieve_convergent_evidence(
        query=q,
        all_turns=all_turns,
        retriever=retriever,
        reference_date=q_date,
        base_top_k=60 if is_agg else 45,
    )
    base_limit = 32 if is_agg else 28
    top_sids = {t[0] for t in cov_turns[:base_limit]} | guaranteed_sids
    candidate_turns = [t for t in all_turns if t[0] in top_sids] or cov_turns[:25]
    if is_agg:
        token_budget = min(2200, int(token_budget * 1.5))

    cand_sids = {c[0] for c in candidate_turns}
    cand_records = [rec for rec in all_records if (m := header_sid_re.match(rec.raw_content or "")) and m.group(1) in cand_sids] or all_records[:50]

    bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)

    if is_agg:
        sampled_ev = sampler.sample(q, candidate_turns, max_total_tokens=token_budget)
        ctx = sampler.format_as_context(sampled_ev)
        if bundle.assembled_certificate_block:
            ctx = f"{bundle.assembled_certificate_block}\n\n{ctx}"
    else:
        t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(candidate_turns)}
        anchor_ids = []
        anchor_scores = {}
        for t in cov_turns:
            nid = t_map.get((t[0], t[3]))
            if nid and nid not in anchor_ids:
                anchor_ids.append(nid)
                anchor_scores[nid] = cov_scores.get((t[0], t[3]), 1.0)
        for i, t in enumerate(candidate_turns):
            nid = f"{t[0]}:t{i}"
            if nid not in anchor_scores and (t[0], t[3]) in cov_scores:
                anchor_scores[nid] = cov_scores[(t[0], t[3])]

        edg = graph_builder.build(candidate_turns, cand_records)
        bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
        compiled = compiler.compile(
            question=q,
            bundles=bundles,
            token_budget=token_budget,
            co_processor_bundle=bundle,
            anchor_scores=anchor_scores,
            is_multi_session=True,
            adaptive_escalation=True,
        )
        ctx = compiled.context_text

    # Check which gold SIDs appear in ctx
    ctx_sids = set()
    for line in ctx.splitlines():
        m = header_sid_re.match(line)
        if m:
            ctx_sids.add(m.group(1))

    # Also for agg path, check content overlap since agg doesn't use [sid] header
    if is_agg:
        ctx_lower = ctx.lower()
        for s_idx, session in enumerate(item.get("haystack_sessions", [])):
            sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
            if sid in gold_sids:
                for turn in session:
                    content = turn.get("content", "").strip()
                    if len(content) > 20 and content[:30].lower() in ctx_lower:
                        ctx_sids.add(sid)
                        break

    missing_in_ctx = gold_sids - ctx_sids
    has_cert = bool(bundle.assembled_certificate_block or bundle.deterministic_answer_override)
    print(f"[{idx+1:02d}] {qid} | mode={'AGG' if is_agg else 'GRAPH'} | gold={len(gold_sids)} | ctx_gold={len(ctx_sids & gold_sids)} | missing={missing_in_ctx} | cert={has_cert}")
