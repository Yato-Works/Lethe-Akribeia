import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
dev_path = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"

with open(dev_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

q_map = {item["question_id"]: item for item in dev_data}

target_qids = [
    # Knowledge Update (5)
    "07741c45", "eace081b", "6aeb4375", "0f05491a", "69fee5aa",
    # Multi-Session Constraints (5)
    "6d550036", "00ca467f", "60472f9c", "gpt4_2f8be40d", "b5ef892d"
]

for qid in target_qids:
    item = q_map.get(qid)
    if not item:
        continue
    print(f"\n{'='*75}")
    print(f"[{qid}] Category: {item.get('question_type')}")
    print(f"Q: {item.get('question')}")
    print(f"Q Date: {item.get('question_date')}")
    print(f"GT: {item.get('answer')}")
    
    ans_sids = set(item.get("answer_session_ids", []))
    hay_sids = item.get("haystack_session_ids", [])
    hay_dates = item.get("haystack_dates", [])
    hay_sessions = item.get("haystack_sessions", [])
    
    for sid, sdate, sess in zip(hay_sids, hay_dates, hay_sessions):
        if sid in ans_sids:
            print(f"\n  --- Answer Session [{sid}] | Date: {sdate} | Turns: {len(sess)} ---")
            for m in sess:
                role = m.get("role", "")
                c = m.get("content", "").strip()
                if len(c) > 300:
                    c = c[:150] + " ... " + c[-100:]
                print(f"    [{role}]: {c}")
