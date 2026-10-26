"""Probe Multi-Session Coverage Engine effect on gold session retrieval (LLM-free)."""

import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.coverage.coverage_engine import MultiSessionCoverageEngine
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor

import importlib.util
spec = importlib.util.spec_from_file_location("eval_v4", str(REPO / "scripts" / "benchmarks" / "eval_v4_pipeline_dev.py"))
eval_v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_v4)
retriever = eval_v4.V4CandidateRetriever()

coverage_engine = MultiSessionCoverageEngine()
extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = {it["question_id"]: it for it in json.load(f)}

r6_path = REPO / "benchmark_results" / "pipeline_v4" / "lme_dev250_v4_R6.json"
r6_data = json.loads(r6_path.read_text(encoding="utf-8"))
fails = [r for r in r6_data["results"] if r["question_type"] == "multi-session" and not r["is_correct"]]

print(f"Testing {len(fails)} multi-session failures with Coverage Engine...")

recovered_all = 0
recovered_partial = 0
remained_miss = 0

for idx, r in enumerate(fails):
    qid = r["question_id"]
    item = all_items[qid]
    q = item["question"]
    q_date = item.get("question_date", "")
    gold_sids = set(item.get("answer_session_ids", []))

    all_turns = []
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        for turn in session:
            speaker = turn.get("role", "user")
            content = turn.get("content", "")
            all_turns.append((sid, s_date, speaker, content))

    plan = planner.plan(q, [], reference_date=q_date)
    is_aggregation = bool(plan.dominant_mode == "aggregation")

    # Baseline retrieval
    base_turns, _ = retriever.rank_turns(q, all_turns, top_k=60 if is_aggregation else 45, reference_date=q_date)
    base_sids = {t[0] for t in base_turns[:30 if is_aggregation else 26]}
    base_missed = gold_sids - base_sids

    # Coverage Engine retrieval
    cov_turns, cov_scores, guaranteed_sids = coverage_engine.retrieve_convergent_evidence(
        query=q,
        all_turns=all_turns,
        retriever=retriever,
        reference_date=q_date,
        base_top_k=60 if is_aggregation else 45,
    )
    cov_sids = {t[0] for t in cov_turns[:32 if is_aggregation else 28]} | guaranteed_sids
    cov_missed = gold_sids - cov_sids

    if len(base_missed) > 0 and len(cov_missed) == 0:
        status = "FULL_RECOVERY (100% gold recovered!)"
        recovered_all += 1
    elif len(cov_missed) < len(base_missed):
        status = f"PARTIAL_RECOVERY (missed {len(base_missed)} -> {len(cov_missed)})"
        recovered_partial += 1
    elif len(cov_missed) == 0 and len(base_missed) == 0:
        status = "ALREADY_PRESENT_IN_CANDIDATES"
    else:
        status = f"STILL_MISSING ({len(cov_missed)}/{len(gold_sids)} gold sids missing)"
        remained_miss += 1

    print(f"[{idx+1:02d}] {qid}: {status}")
    if len(base_missed) > 0:
        print(f"     Base missed: {base_missed} | Cov missed: {cov_missed}")

print("=" * 60)
print(f"Summary: Full Recovery: {recovered_all} | Partial Recovery: {recovered_partial} | Still Missing: {remained_miss}")
