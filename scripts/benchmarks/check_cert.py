"""LLM-free certificate behavior check for any qid (generalized verify_durcert)."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qids", required=True)
    args = parser.parse_args()

    with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json",
              encoding="utf-8") as f:
        items = json.load(f)

    extractor = UniversalIRExtractor()
    planner = CognitiveExecutionPlanner()
    coprocessor = CoProcessorEngine()

    for qid in [s.strip() for s in args.qids.split(",") if s.strip()]:
        item = next(x for x in items if x["question_id"] == qid)
        q = item["question"]
        q_date = item.get("question_date", "")
        all_records = []
        dates = item.get("haystack_dates", [])
        sids = item.get("haystack_session_ids", [])
        for s_idx, session in enumerate(item.get("haystack_sessions", [])):
            s_date = dates[s_idx] if s_idx < len(dates) else ""
            sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
            for turn in session:
                line = f"[{sid} on {s_date}] {turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
                for r in extractor.extract(turn.get("content", ""),
                                           default_source=turn.get("role", "user")):
                    r.raw_content = line
                    r.time_scope = s_date
                    all_records.append(r)
        plan = planner.plan(q, all_records, reference_date=q_date)
        bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)
        print(f"=== {qid} ===")
        print(f"Q: {q}")
        print(f"GT: {item.get('answer')}")
        print(f"mode={plan.dominant_mode} agg={plan.aggregation_demand}")
        print(f"dur_cert={bundle.aggregation_cert.has_certificate} "
              f"type={bundle.aggregation_cert.query_type} "
              f"total={bundle.aggregation_cert.total_count}")
        if bundle.aggregation_cert.has_certificate:
            print(bundle.aggregation_cert.formatted_certificate[:1200])
        print()


if __name__ == "__main__":
    main()
