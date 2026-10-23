import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = {x["question_id"]: x for x in json.load(f)}

for qid in ("157a136e", "c18a7dc8"):
    it = items[qid]
    q = it["question"]
    q_date = it.get("question_date", "")

    extractor = UniversalIRExtractor()
    records = []
    for s_idx, sess in enumerate(it["haystack_sessions"]):
        sid = it["haystack_session_ids"][s_idx]
        s_date = it["haystack_dates"][s_idx]
        for turn in sess:
            for rec in extractor.extract(turn.get("content"), default_source=turn.get("role")):
                rec.raw_content = f"[{sid}] {turn.get('role')}: {turn.get('content')}"
                rec.time_scope = s_date
                records.append(rec)

    planner = CognitiveExecutionPlanner()
    plan = planner.plan(q, records, reference_date=q_date)

    coproc = CoProcessorEngine()
    bundle = coproc.process(q, records, plan, reference_date=q_date)

    print(f"\n=== QID: {qid} ({q}) ===")
    print("override_answer:", repr(bundle.deterministic_answer_override))
    print("a_cert has_cert:", bundle.aggregation_cert.has_certificate)
    print("a_cert direct:", bundle.aggregation_cert.direct_answer_candidate)
    print("a_cert formatted:\n", bundle.aggregation_cert.formatted_certificate)
