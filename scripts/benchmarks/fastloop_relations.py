import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.certificates.coprocessor_engine import CoProcessorEngine
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor

dev_path = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"
with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

q_map = {item["question_id"]: item for item in dev_data}

target_qids = [
    # Knowledge Update
    "07741c45", "eace081b", "6aeb4375", "0f05491a", "69fee5aa",
    # Multi-Session Constraints
    "60472f9c", "00ca467f", "b5ef892d", "gpt4_2f8be40d"
]

extractor = UniversalIRExtractor()
planner = CognitiveExecutionPlanner()
coprocessor = CoProcessorEngine()

print("=" * 75)
print(f"FASTLOOP EVALUATION ON 9 TARGET RELATION QUESTIONS")
print("=" * 75)

for qid in target_qids:
    item = q_map.get(qid)
    if not item:
        continue
    q = item["question"]
    q_date = item.get("question_date", "")
    gt = str(item.get("answer", ""))
    cat = item.get("question_type", "")

    # Extract all records from haystack sessions
    all_records = []
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    for s_idx, session in enumerate(item.get("haystack_sessions", [])):
        s_date = dates[s_idx] if s_idx < len(dates) else ""
        sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
        for turn_idx, turn in enumerate(session):
            speaker = turn.get("role", "user")
            content = turn.get("content", "")
            line = f"[{sid} on {s_date}] {speaker.capitalize()}: {content}"
            recs = extractor.extract(content, default_source=speaker)
            for r in recs:
                r.raw_content = line
                r.time_scope = s_date
                r.metadata = {"session_id": sid, "turn_idx": turn_idx}
                all_records.append(r)

    plan = planner.plan(q, all_records, reference_date=q_date)
    bundle = coprocessor.process(q, all_records, plan, reference_date=q_date)

    override = bundle.deterministic_answer_override
    has_cert = bool(bundle.assembled_certificate_block)

    print(f"\n[{qid}] ({cat}) Q: {q[:70]}")
    print(f"  GT:       {gt[:50]}")
    print(f"  OVERRIDE: {override}")
    print(f"  HAS_CERT: {has_cert}")
    if has_cert:
        # print first 2 lines of cert
        cert_preview = bundle.assembled_certificate_block.split("\n")[:3]
        print(f"  CERT:     {' | '.join(cert_preview)}")
