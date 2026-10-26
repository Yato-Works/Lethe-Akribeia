"""LLM-free failure anatomy: for every failed multi-session + knowledge-update
question, rebuild the V4 pipeline context (no reader call) and check:
  - which path (aggregation / graph-compile) was taken
  - whether the ground-truth answer string/number appears in the compiled context
  - whether the gold answer sessions made it into candidates / selection

Run: python scripts/benchmarks/diag_fail_anatomy.py
"""
import importlib.util
import json
import os
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
    "eval_v4_pipeline_dev",
    str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
retriever = mod.V4CandidateRetriever()

extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()
graph_builder = GraphBuilder()
compiler = BudgetedCognitiveCompiler()
coprocessor = CoProcessorEngine()
sampler = PerSessionSampler(turns_per_session=2)

RESULTS = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_pipeline.json"
data = json.loads(RESULTS.read_text(encoding="utf-8"))
res = data["results"] if isinstance(data, dict) and "results" in data else data

TARGET_TYPES = (
    {"multi-session", "knowledge-update"}
    if os.environ.get("ALL_TYPES") != "1"
    else None  # None -> every failed question
)
fails = [r for r in res
         if (TARGET_TYPES is None or r["question_type"] in TARGET_TYPES)
         and not r["is_correct"]]

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = json.load(f)
by_id = {x["question_id"]: x for x in all_items}


def gt_variants(gt: str) -> list[str]:
    """Extract checkable answer tokens (numbers, money, short strings)."""
    gt = str(gt).strip()
    out = set()
    if gt:
        out.add(gt.lower())
    for m in re.findall(r"\d[\d,]*", gt):
        out.add(m.replace(",", "").lower())
    return list(out)


rows = []
for r in fails:
    qid = r["question_id"]
    item = by_id.get(qid)
    if not item:
        continue
    q = item["question"]
    q_date = item.get("question_date", "")
    gt = str(r.get("ground_truth", ""))

    all_turns, all_records = [], []
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    gold_sids = set(item.get("answer_session_ids", []) or [])
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
    token_budget = plan.recommended_budget
    is_agg = bool(plan.dominant_mode == "aggregation")

    # 3. Candidate Retrieval (mirror eval_v4_pipeline_dev.py exactly)
    if is_agg:
        top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=60, reference_date=q_date)
        top_sids = {t[0] for t in top_turns[:30]}
        candidate_turns = [t for t in all_turns if t[0] in top_sids] or list(all_turns)
        token_budget = min(2200, int(token_budget * 1.5))
    else:
        top_turns, turn_scores = retriever.rank_turns(q, all_turns, top_k=35, reference_date=q_date)
        top_sids = {t[0] for t in top_turns[:18]}
        candidate_turns = [t for t in all_turns if t[0] in top_sids] or top_turns[:15]

    cand_sids = {c[0] for c in candidate_turns}
    header_sid_re = re.compile(r"^\[([^\s\]]+)")
    cand_records = []
    for rec in all_records:
        m = header_sid_re.match(rec.raw_content or "")
        if m and m.group(1) in cand_sids:
            cand_records.append(rec)
    if not cand_records:
        cand_records = all_records[:50]

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

    # 4-6. Co-processor + path selection + compiled context (no reader call)
    bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)
    det_pred = ""
    if bundle.deterministic_answer_override:
        path = "det"
        det_pred = bundle.deterministic_answer_override
        ctx = "[0-TOKEN DETERMINISTIC RESOLUTION]"
    elif is_agg:
        path = "agg"
        sampled_ev = sampler.sample(q, candidate_turns, max_total_tokens=token_budget)
        ctx = sampler.format_as_context(sampled_ev)
        if bundle.assembled_certificate_block:
            ctx = f"{bundle.assembled_certificate_block}\n\n{ctx}"
    else:
        path = "graph"
        edg = graph_builder.build(candidate_turns, cand_records)
        bundles = edg.form_bundles(anchor_ids, max_bundle_tokens=150)
        is_multi = bool(
            plan.requires_event_aggregation
            or plan.aggregation_demand >= 0.50
            or plan.temporal_demand >= 0.85
        )
        compiled = compiler.compile(
            question=q,
            bundles=bundles,
            token_budget=token_budget,
            co_processor_bundle=bundle,
            anchor_scores=anchor_scores,
            is_multi_session=is_multi,
        )
        ctx = compiled.context_text

    # 7. LLM-free checks
    ctx_l = (ctx or "").lower()
    gt_in_ctx = any(v and v in ctx_l for v in gt_variants(gt))
    # Stricter: GT variant must appear in a line belonging to a gold session
    # (guards single-digit GTs like "3" matching anywhere in the context).
    gt_in_gold_line = False
    if gt_in_ctx and gold_sids:
        for line in ctx_l.splitlines():
            m = header_sid_re.match(line)
            if m and m.group(1) in gold_sids:
                if any(v and v in line for v in gt_variants(gt)):
                    gt_in_gold_line = True
                    break
    dump_qid = os.environ.get("DUMP_QID", "")
    if dump_qid and qid == dump_qid:
        print(f"\n----- CTX DUMP {qid} (path={path}, gt={gt!r}) -----")
        print(ctx)
        print("----- END CTX -----\n")

    # Gold-turn presence probe: does ANY gold-session turn survive verbatim
    # into the compiled context? (Works for both paths: agg keeps turn text
    # verbatim (no sid headers), graph keeps "[sid on date] ..." headers.)
    gold_turn_texts = []
    gt_var = [v for v in gt_variants(gt) if v]
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        if sid in gold_sids:
            for turn in session:
                gold_turn_texts.append(turn.get("content", "").lower())

    def _present(text: str) -> bool:
        probe = text[:40]
        return len(probe) >= 15 and probe in ctx_l

    gold_turns_in_ctx = sum(1 for t in gold_turn_texts if _present(t))
    # STRICT: turns that actually carry the GT value must be in the context.
    gt_bearing = [t for t in gold_turn_texts if any(v in t for v in gt_var)]
    gt_bearing_in_ctx = sum(1 for t in gt_bearing if _present(t))
    gold_in_cand = bool(gold_sids) and bool(gold_sids & cand_sids)
    gold_in_top = bool(gold_sids) and bool(gold_sids & top_sids)
    det_correct = ""
    if det_pred:
        det_correct = "Y" if any(v and v in str(det_pred).lower() for v in gt_variants(gt)) else "N"

    if not gold_sids:
        bucket = "no_gold_sid"
    elif not gold_in_cand:
        bucket = "retrieval_miss"        # gold session never entered candidates
    elif not gt_in_ctx and gt_bearing:
        # A gold turn carrying the GT exists in the haystack:
        # answer-bearing evidence was dropped by compilation/selection.
        bucket = (
            "selection_miss" if gt_bearing_in_ctx == 0
            else "partial_gt_bearing"     # some bearing turns in, GT still absent
        )
    elif not gt_in_ctx:
        bucket = "computed_gt_unverifiable"  # GT is computed/paraphrased; gold turns in
    elif det_pred and det_correct == "N":
        bucket = "committer_bug"          # deterministic override claimed wrong ans
    elif len(gt.strip()) <= 4 and gt_bearing and gt_bearing_in_ctx == 0:
        # Short numeric/yes-no GT: the verbatim hit may be a date/number
        # coincidence ("3" matches "2023"), and no GT-bearing gold turn was
        # compiled -> evidence for the answer itself was NOT selected.
        bucket = "reader_miss_unverified"
    else:
        bucket = "reader_miss"            # GT present in context, reader wrong

    rows.append({
        "qid": qid, "qtype": r["question_type"][:12], "path": path,
        "bucket": bucket, "gt_in_ctx": gt_in_ctx,
        "gold_in_cand": gold_in_cand, "gold_in_top": gold_in_top,
        "n_gold": len(gold_sids), "n_gold_cand": len(gold_sids & cand_sids),
        "gold_turns_in_ctx": gold_turns_in_ctx,
        "n_gold_turns": len(gold_turn_texts),
        "n_gt_bearing": len(gt_bearing),
        "n_gt_bearing_in_ctx": gt_bearing_in_ctx,
        "ctx_chars": len(ctx or ""), "gt": gt[:40], "pred": r["predicted_answer"][:40],
    })

# ---- report ----
from collections import Counter

print(f"\n=== LLM-free failure anatomy: {len(rows)}/{len(fails)} failed "
      f"multi-session + knowledge-update questions analyzed ===\n")
hdr = f"{'qid':<10} {'type':<13} {'path':<6} {'bucket':<23} {'gt?':<4} {'gold':<6} {'gturns':<9} {'gtbear(in)':<11} {'ctx':>6}  {'GT':<40} {'PRED':<40}"
print(hdr)
print("-" * len(hdr))
for row in rows:
    print(f"{row['qid']:<10} {row['qtype']:<13} {row['path']:<6} {row['bucket']:<23} "
          f"{('Y' if row['gt_in_ctx'] else 'N'):<4} "
          f"{str(row['n_gold_cand'])+'/'+str(row['n_gold']):<6} "
          f"{str(row['gold_turns_in_ctx'])+'/'+str(row['n_gold_turns']):<9} "
          f"{str(row['n_gt_bearing_in_ctx'])+'/'+str(row['n_gt_bearing']):<11} "
          f"{row['ctx_chars']:>6}  "
          f"{row['gt']:<40} {row['pred']:<40}")

print("\n=== bucket counts (all analyzed fails) ===")
for k, v in Counter(r["bucket"] for r in rows).most_common():
    print(f"  {k:<26} {v:>3}  ({v/len(rows)*100:.0f}%)")
print("\n=== path x bucket ===")
paths = sorted({r["path"] for r in rows})
buckets = sorted({r["bucket"] for r in rows})
print(f"{'bucket':<26}" + "".join(f"{p:>8}" for p in paths))
for b in buckets:
    print(f"{b:<26}" + "".join(
        f"{sum(1 for r in rows if r['bucket']==b and r['path']==p):>8}" for p in paths))
print("\n=== question_type x bucket ===")
types = sorted({r["qtype"] for r in rows})
print(f"{'type':<14}" + "".join(f"{b[:11]:>13}" for b in buckets))
for t in types:
    print(f"{t:<14}" + "".join(
        f"{sum(1 for r in rows if r['bucket']==b and r['qtype']==t):>13}" for b in buckets))
print("\n=== GT present in compiled context (answerable iff reader fails) ===")
in_ctx = sum(1 for r in rows if r["gt_in_ctx"])
print(f"  gt_in_ctx = {in_ctx}/{len(rows)} ({in_ctx/len(rows)*100:.0f}%)"
      f"  -> reader_miss ceiling; missing = retrieval/selection bottleneck")
