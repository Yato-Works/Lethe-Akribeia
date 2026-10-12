import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
dev_path = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"

with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

q_map = {item["question_id"]: item for item in dev_data}

target_ms = ["6d550036"]

for qid in target_ms:
    item = q_map.get(qid)
    print(f"\n{'='*70}\n[{qid}] Q: {item['question']}\nGT: {item['answer']}\nQ Date: {item['question_date']}")
    sids = set(item["answer_session_ids"])
    for sid, d, s in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]):
        if sid in sids:
            print(f"\n--- SID: {sid} | Date: {d} ---")
            for m in s:
                if m["role"] == "user":
                    print(f"  [user]: {m['content']}")
