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

def diag(qid):
    it = next(item for item in data if item["question_id"] == qid)
    records = []
    for s_idx, session in enumerate(it.get("haystack_sessions", [])):
        sid = it["haystack_session_ids"][s_idx]
        s_date = it["haystack_dates"][s_idx]
        for turn in session:
            speaker = turn.get("role", "user")
            content = turn.get("content", "")
            header = f"[{sid} on {s_date}]"
            line = f"{header} {speaker.capitalize()}: {content}"
            for r in extractor.extract(content, default_source=speaker):
                r.raw_content = line
                r.time_scope = s_date
                records.append(r)

    cert = engine.generate(it["question"], records)
    print("=" * 60)
    print(f"QID: {qid} | Question: {it['question']}")
    print(f"Cert:\n{cert.formatted_certificate}")

diag('edced276')
diag('f35224e0')
diag('a3332713')
diag('a1cc6108')
