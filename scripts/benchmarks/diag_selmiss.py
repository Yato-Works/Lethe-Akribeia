"""Where exactly do GT-bearing turns die? Stage-by-stage autopsy (LLM-free).

For each known selection_miss question, the GT-bearing gold turns are
tracked through the V4 pipeline:
  retrieval rank/score -> candidate set -> anchor set -> bundle -> compiled ctx

Run: python scripts/benchmarks/diag_selmiss.py
"""

from __future__ import annotations

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

spec = importlib.util.spec_from_file_location(
    "ev", str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

QIDS = ["36b9f61e", "9ee3ecd6", "gpt4_b5700ca0", "gpt4_5dcc0aab",
        "gpt4_2f56ae70", "eace081b", "3249768e", "eaca4986"]

extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()
retriever = ev.V4CandidateRetriever()
graph_builder = GraphBuilder()
compiler = BudgetedCognitiveCompiler()
coprocessor = CoProcessorEngine()
sampler = PerSessionSampler(turns_per_session=3)

items = json.loads((REPO / "datasets" / "external" / "longmemeval_s_cleaned.json")
                   .read_text(encoding="utf-8"))
by_id = {x["question_id"]: x for x in items}


def gt_variants(gt: str) -> list[str]:
    out = {str(gt).strip().lower()}
    for m in re.findall(r"\d[\d,]*", str(gt)):
        out.add(m.replace(",", ""))
    return [v for v in out if v]


for qid in QIDS:
    item = by_id[qid]
    gt = str(item.get("answer", ""))
    variants = gt_variants(gt)
    gold_sids = set(item.get("answer_session_ids") or [])
    q = item["question"]
    q_date = item.get("question_date", "")

    all_turns, all_records = [], []
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        for turn in session:
            speaker = turn.get("role", "user")
            content = turn.get("content", "")
            header = f"[{sid} on {s_date}]" if s_date else f"[{sid}]"
            all_turns.append((sid, s_date, speaker, content))
            for rec in extractor.extract(content, default_source=speaker):
                rec.raw_content = f"{header} {speaker.capitalize()}: {content}"
                rec.time_scope = s_date
                all_records.append(rec)

    plan = planner.plan(q, all_records, reference_date=q_date)
    is_agg = bool(plan.dominant_mode == "aggregation")
    if is_agg:
        top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=60, reference_date=q_date)
        top_sids = {t[0] for t in top_turns[:30]}
        candidate_turns = [t for t in all_turns if t[0] in top_sids] or list(all_turns)
        token_budget = min(2200, int(plan.recommended_budget * 1.5))
    else:
        top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=35, reference_date=q_date)
        top_sids = {t[0] for t in top_turns[:18]}
        candidate_turns = [t for t in all_turns if t[0] in top_sids] or top_turns[:15]
        token_budget = plan.recommended_budget

    cand_sids = {c[0] for c in candidate_turns}
    header_sid_re = re.compile(r"^\[([^\s\]]+)")
    cand_records = [r for r in all_records
                    if (m := header_sid_re.match(r.raw_content or ""))
                    and m.group(1) in cand_sids]

    t_map = {(t[0], t[3]): f"{t[0]}:t{i}" for i, t in enumerate(candidate_turns)}
    anchor_ids, anchor_scores = [], {}
    for t in top_turns:
        nid = t_map.get((t[0], t[3]))
        if nid and nid not in anchor_ids:
            anchor_ids.append(nid)
            anchor_scores[nid] = turn_scores.get((t[0], t[3]), 1.0)
    for i, t in enumerate(candidate_turns):
        nid = f"{t[0]}:t{i}"
        if nid not in anchor_scores and (t[0], t[3]) in turn_scores:
            anchor_scores[nid] = turn_scores[(t[0], t[3])]

    bundle_ids: set = set()
    bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)
    if is_agg:
        sampled = sampler.sample(q, candidate_turns, max_total_tokens=token_budget)
        compiled_ctx = sampler.format_as_context(sampled)
        if bundle.assembled_certificate_block:
            compiled_ctx = f"{bundle.assembled_certificate_block}\n\n{compiled_ctx}"
        stage_detail = f"sampler picked {len(sampled)}/{len(candidate_turns)} turns budget={token_budget}"
    else:
        edg = graph_builder.build(candidate_turns, cand_records)
        bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
        for b in bundles:
            bundle_ids.update(n.node_id for n in b.member_nodes)
        compiled = compiler.compile(
            question=q, bundles=bundles, token_budget=token_budget,
            co_processor_bundle=bundle, anchor_scores=anchor_scores,
            is_multi_session=bool(plan.requires_event_aggregation
                                  or plan.aggregation_demand >= 0.50
                                  or plan.temporal_demand >= 0.85))
        compiled_ctx = compiled.context_text
        stage_detail = f"{len(bundles)} bundles budget={token_budget}"
    ctx_l = compiled_ctx.lower()

    # --- track each GT-bearing turn ---
    print("=" * 100)
    print(f"{qid} [{plan.dominant_mode}] gt={gt!r} | {stage_detail}")

    # Knapsack trace (graph path): why was the bearing bundle not selected?
    if not is_agg:
        cert_tokens = (len(bundle.assembled_certificate_block.split()) * 4 // 3
                       if bundle.assembled_certificate_block else 0)
        eff = max(150, token_budget - cert_tokens)
        bearing_keys = {(sid, content) for (sid, _d, _sp, content) in all_turns
                        if sid in gold_sids and any(v in content.lower() for v in variants)}
        bearing_nids = {t_map[k] for k in bearing_keys if k in t_map}
        print(f"  budget={token_budget} cert_tokens={cert_tokens} effective={eff} "
              f"bundles={len(bundles)}")
        sel = set(compiled.selected_bundle_ids)
        for b in bundles:
            b_nids = {n.node_id for n in b.member_nodes}
            if not (b_nids & bearing_nids):
                continue
            max_anc = max((anchor_scores.get(n.node_id, 0.0) for n in b.member_nodes),
                          default=0.0)
            print(f"  BEARING-BUNDLE {b.bundle_id}: tokens={b.total_tokens} "
                  f"max_anchor={max_anc:.1f} valid={b.total_tokens <= eff} "
                  f"SELECTED={b.bundle_id in sel} rationale={b.rationale[:50]!r}")

    rank_of = {(t[0], t[3]): i + 1 for i, t in enumerate(top_turns)}
    cand_keys = {(c[0], c[3]) for c in candidate_turns}
    n_bearing = 0
    for key in ((sid, _d, sp, content) for (sid, _d, sp, content) in all_turns):
        sid, _date, _speaker, content = key
        if sid not in gold_sids:
            continue
        if not any(v in content.lower() for v in variants):
            continue
        n_bearing += 1
        k2 = (sid, content)
        rank = rank_of.get(k2)
        nid = t_map.get(k2)
        in_cand = k2 in cand_keys
        is_anchor = bool(nid) and nid in anchor_ids
        in_bundle = (nid in bundle_ids) if (nid and not is_agg) else None
        in_ctx = content[:60].lower() in ctx_l
        if not in_cand:
            flag = "LOST@candidates"
        elif rank is None:
            flag = "LOST@retrieval(top_k)"
        elif not is_anchor:
            flag = "LOST@anchor"
        elif in_bundle is False:
            flag = "LOST@bundle"
        elif not in_ctx:
            flag = "LOST@compile(budget)"
        else:
            flag = "KEPT"
        print(f"  [{flag:<26}] rank={str(rank):<4} score="
              f"{turn_scores.get(k2, 0.0):<7.2f} sid={sid} | {content[:60]!r}")
    if n_bearing == 0:
        print("  (no bearing turn matched by GT variants -- paraphrased GT)")

