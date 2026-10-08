"""LLM-free verification of the duration-sum certificate on 6cb6f249."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json",
          encoding="utf-8") as f:
    items = json.load(f)
item = next(x for x in items if x["question_id"] == "6cb6f249")
q = item["question"]
q_date = item.get("question_date", "")

extractor = UniversalIRExtractor()
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

planner = CognitiveExecutionPlanner()
plan = planner.plan(q, all_records, reference_date=q_date)
print(f"dominant_mode={plan.dominant_mode} agg={plan.aggregation_demand} "
      f"requires_event_agg={plan.requires_event_aggregation}")

bundle = CoProcessorEngine().process(q, all_records, plan, reference_date=q_date)
print(f"has_cert={bundle.aggregation_cert.has_certificate} "
      f"type={bundle.aggregation_cert.query_type}")
print(bundle.aggregation_cert.formatted_certificate[:1500])
