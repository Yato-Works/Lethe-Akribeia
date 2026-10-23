"""Diagnose knowledge-update regressions: 0f05491a, eace081b, affe2881.

Runs the eval_v4_pipeline_dev compile path (no LLM) and reports:
- planner mode / budget
- whether gold-answer evidence survives retrieval + selection
- compiled context content around key answer terms

Run: python scripts/benchmarks/diag_ku_regression.py
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from artificial_memory_gen.compiler.budgeted_compiler import BudgetedCognitiveCompiler
from artificial_memory_gen.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.graph.graph_builder import GraphBuilder
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory_gen.compiler.samplers.per_session_sampler import PerSessionSampler


def load_eval():
    spec = importlib.util.spec_from_file_location(
        "eval_v4_pipeline_dev",
        str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QIDS = ["0f05491a", "eace081b", "affe2881"]
KEY_TERMS = {
    "0f05491a": ["120", "gold level", "starbucks"],
    "eace081b": ["oahu"],
    "affe2881": ["32"],
}

mod = load_eval()
retriever = mod.V4CandidateRetriever()
extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()
graph_builder = GraphBuilder()
compiler = BudgetedCognitiveCompiler()
coprocessor = CoProcessorEngine()
sampler = PerSessionSampler(turns_per_session=2)

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = json.load(f)

for qid in QIDS:
    item = next(x for x in all_items if x["question_id"] == qid)
    q = item["question"]
    q_date = item.get("question_date", "")
    gt = str(item.get("answer", ""))
    print("=" * 90)
    print(f"{qid}: {q}  (GT={gt})")

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
            for r in extractor.extract(content, default_source=speaker):
                r.raw_content = line
                r.time_scope = s_date
                all_records.append(r)

    plan = planner.plan(q, all_records, reference_date=q_date)
    token_budget = plan.recommended_budget
    is_agg = bool(plan.dominant_mode == "aggregation")
    print(f"  dominant_mode={plan.dominant_mode} agg={is_agg} budget={token_budget} "
          f"agg_demand={plan.aggregation_demand:.2f} temp_demand={plan.temporal_demand:.2f}")

    if is_agg:
        top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=60, reference_date=q_date)
        top_sids = {t[0] for t in top_turns[:30]}
        candidate_turns = [t for t in all_turns if t[0] in top_sids] or list(all_turns)
        token_budget = min(2200, int(token_budget * 1.5))
    else:
        top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=35, reference_date=q_date)
        top_sids = {t[0] for t in top_turns[:18]}
        candidate_turns = [t for t in all_turns if t[0] in top_sids]
        if not candidate_turns:
            candidate_turns = top_turns[:15]

    # Which sessions contain key answer terms (ground truth source)?
    keys = KEY_TERMS.get(qid, [])
    src_sids = set()
    for sid, _, _, content in all_turns:
        low = content.lower()
        if any(k in low for k in keys):
            src_sids.add(sid)
    print(f"  sessions containing key terms {keys}: {sorted(src_sids)}")
    print(f"  candidate sessions ({len(top_sids)}): {sorted(top_sids)}")
    print(f"  MISSED BY RETRIEVER: {sorted(src_sids - top_sids)}")

    bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)
    print(f"  deterministic_override={bundle.deterministic_answer_override!r}")
    if bundle.assembled_certificate_block:
        print("  --- certificate block ---")
        for ln in bundle.assembled_certificate_block.splitlines()[:20]:
            print("   |", ln[:160])

    if bundle.deterministic_answer_override:
        continue

    if is_agg:
        sampled = sampler.sample(q, candidate_turns, max_total_tokens=token_budget)
        ctx = sampler.format_as_context(sampled)
        sel_sids = {e.session_id for e in sampled}
        if bundle.assembled_certificate_block:
            ctx = f"{bundle.assembled_certificate_block}\n\n{ctx}"
        print(f"  selected sessions ({len(sel_sids)}): {sorted(sel_sids)}")
        print(f"  MISSED BY SAMPLER: {sorted(src_sids - sel_sids)}")
    else:
        cand_sids = {c[0] for c in candidate_turns}
        header_sid_re = re.compile(r"^\[([^\s\]]+)")
        cand_records = []
        for r in all_records:
            m = header_sid_re.match(r.raw_content or "")
            if m and m.group(1) in cand_sids:
                cand_records.append(r)
        if not cand_records:
            cand_records = all_records[:50]
        t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(candidate_turns)}
        anchor_ids = []
        anchor_scores = {}
        for t in top_turns:
            nid = t_map.get((t[0], t[3]))
            if nid and nid not in anchor_ids:
                anchor_ids.append(nid)
                anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)
        for i, t in enumerate(candidate_turns):
            nid = f"{t[0]}:t{i}"
            if nid not in anchor_scores and (t[0], t[3]) in turn_scores:
                anchor_scores[nid] = turn_scores[(t[0], t[3])]
        edg = graph_builder.build(candidate_turns, cand_records)
        bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
        is_multi = bool(plan.requires_event_aggregation or plan.aggregation_demand >= 0.50 or plan.temporal_demand >= 0.85)
        compiled = compiler.compile(
            question=q, bundles=bundles, token_budget=token_budget,
            co_processor_bundle=bundle, anchor_scores=anchor_scores, is_multi_session=is_multi,
        )
        ctx = compiled.context_text

    low = ctx.lower()
    print("  --- context key-term presence ---")
    for k in keys:
        print(f"    {k!r}: {'FOUND' if k in low else 'MISSING'}")
    print(f"  context length: {len(ctx.split())} words / ~{len(ctx)//4} tokens")
    for k in keys:
        i = low.find(k)
        if i >= 0:
            print(f"    ctx[{k}]: ...{ctx[max(0, i-140):i+140].replace(chr(10), ' | ')}...")
