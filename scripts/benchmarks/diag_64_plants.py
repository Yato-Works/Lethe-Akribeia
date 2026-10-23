import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory_gen.compiler.certificates.aggregation_certificate import AggregationCertificateEngine
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor

data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
extractor = UniversalIRExtractor()
engine = AggregationCertificateEngine()

# 6456829e
it_64 = next(it for it in data if it["question_id"] == "6456829e")
records_64 = []
for s_idx, session in enumerate(it_64.get("haystack_sessions", [])):
    sid = it_64["haystack_session_ids"][s_idx]
    s_date = it_64["haystack_dates"][s_idx]
    for turn in session:
        line = f"[{sid} on {s_date}] {turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
        for r in extractor.extract(turn.get('content', ''), default_source=turn.get('role', 'user')):
            r.raw_content = line
            records_64.append(r)

v_a, ev_a = engine._extract_entity_turn_val("tomatoes", records_64, "plants")
v_b, ev_b = engine._extract_entity_turn_val("cucumbers", records_64, "plants")
print("6456829e Tomatoes:", v_a, ev_a)
print("6456829e Cucumbers:", v_b, ev_b)
