"""Diagnose why duration-sum cert does not fire on 6cb6f249."""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from artificial_memory.compiler.certificates.aggregation_certificate import AggregationCertificateEngine
from artificial_memory.compiler.ir_extractor import UniversalIRExtractor

with open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json",
          encoding="utf-8") as f:
    items = json.load(f)
item = next(x for x in items if x["question_id"] == "6cb6f249")
q = item["question"]

print("Q regex match:", bool(AggregationCertificateEngine._RE_DURATION_Q.search(q)))

extractor = UniversalIRExtractor()
all_records = []
dates = item.get("haystack_dates", [])
sids = item.get("haystack_session_ids", [])
n_turns = 0
for s_idx, session in enumerate(item.get("haystack_sessions", [])):
    s_date = dates[s_idx] if s_idx < len(dates) else ""
    sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
    for turn in session:
        n_turns += 1
        line = f"[{sid} on {s_date}] {turn.get('role', 'user').capitalize()}: {turn.get('content', '')}"
        for r in extractor.extract(turn.get("content", ""),
                                   default_source=turn.get("role", "user")):
            r.raw_content = line
            r.time_scope = s_date
            all_records.append(r)
print(f"turns={n_turns} records={len(all_records)}")

pat = re.compile(r"(10-day|week-long|\d+\s*days?\s+break)", re.I)
rec_hits = [r.raw_content[:150] for r in all_records if pat.search(r.raw_content or "")]
print(f"records containing duration-break: {len(rec_hits)}")
for h in rec_hits[:10]:
    print("  -", h)

eng = AggregationCertificateEngine()
cert = eng._resolve_duration_sum(q, all_records)
print("cert fires:", cert.has_certificate, cert.query_type, cert.total_count)
