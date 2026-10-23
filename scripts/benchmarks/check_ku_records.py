import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
dev_path = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"

sys.path.insert(0, str(REPO / "src"))
from artificial_memory_gen.compiler.ir_extractor import UniversalIRExtractor

with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

q_map = {item["question_id"]: item for item in dev_data}
extractor = UniversalIRExtractor()

target_ku = ["07741c45", "eace081b", "6aeb4375", "0f05491a", "69fee5aa"]

for qid in target_ku:
    item = q_map.get(qid)
    print(f"\n{'='*75}\n[{qid}] Q: {item['question']}")
    sids = set(item["answer_session_ids"])
    for sid, d, s in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]):
        if sid in sids:
            records = []
            for t_idx, m in enumerate(s):
                role = m.get("role", "user")
                c = m.get("content", "")
                recs = extractor.extract(c, default_source=role)
                for r in recs:
                    r.raw_content = f"[{sid} on {d}] {role.capitalize()}: {c}"
                    r.time_scope = d
                    records.append(r)
            print(f"\n--- SID: {sid} Date: {d} (extracted {len(records)} records) ---")
            for r in records:
                txt = r.raw_content
                if any(w in txt.lower() for w in ("sneaker", "bed", "closet", "rack", "hawaii", "kauai", "oahu", "korean", "restaurant", "starbucks", "star", "coin", "1920")):
                    print(f"  [Turn] text: {txt[:120]}...")
                    print(f"     entity: {r.entity} | prop: {r.property} | val: {r.value} | date: {r.time_scope}")


