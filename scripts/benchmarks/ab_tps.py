"""LLM-free A/B: turns_per_session=2 vs 3 on the 5 duration multi questions.

Measures sampled-turn counts + score coverage only (no reader calls).
Run: python scripts/benchmarks/ab_tps.py
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

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


with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    all_items = json.load(f)
mod = load_eval()
retriever = mod.V4CandidateRetriever()
for qid in ["6cb6f249", "edced276", "b5ef892d", "5a7937c8", "a08a253f"]:
    item = next(x for x in all_items if x["question_id"] == qid)
    q = item["question"]
    q_date = item.get("question_date", "")
    all_turns = []
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        for turn in session:
            all_turns.append((sid, s_date, turn.get("role", "user"), turn.get("content", "")))
    top_turns, _ = retriever.rank_turns(q, all_turns, top_k=60, reference_date=q_date)
    top_sids = {t[0] for t in top_turns[:30]}
    cand = [t for t in all_turns if t[0] in top_sids]
    for tps in (2, 3):
        s = PerSessionSampler(turns_per_session=tps).sample(q, cand, max_total_tokens=2200)
        sess = sorted({e.session_id for e in s})
        print(f"{qid} tps={tps}: {len(s)} turns / {len(sess)} sessions: {sess}")
