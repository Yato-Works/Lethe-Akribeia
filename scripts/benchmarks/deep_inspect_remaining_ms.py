import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

data = json.load(open("benchmark_results/pipeline_v4/lme_dev250_v4_R8.json", encoding="utf-8"))
results = data["results"]
fails = {r["question_id"]: r for r in results if not r.get("is_correct", False)}

target_qids = [
    "0a995998",      # clothing 3
    "gpt4_59c863d7", # model kits 5
    "gpt4_15e38248", # furniture 4
    "4f54b7c9",      # antiques 5
    "gpt4_a56e767c", # movie festivals 4
    "27016adc",      # renovation percentage 10%
    "6d550036",      # projects led 2
]

# Load dev dataset to see all sessions
with open("datasets/external/longmemeval_s_cleaned.json", "r", encoding="utf-8") as f:
    dev_data = json.load(f)

dev_map = {item["question_id"]: item for item in dev_data}

for qid in target_qids:
    if qid not in dev_map:
        continue
    item = dev_map[qid]
    print(f"\n==================== QID: {qid} ====================")
    print(f"QUESTION: {item['question']}")
    print(f"GROUND TRUTH: {item.get('answer', item.get('ground_truth'))}")
    ans_sids = set(item.get("answer_session_ids", []))
    haystack_sessions = item.get("haystack_sessions", [])
    haystack_sids = item.get("haystack_session_ids", [])
    print(f"Total haystack sessions: {len(haystack_sessions)}, Answer session IDs: {ans_sids}")
    for s_idx, (sid, sess) in enumerate(zip(haystack_sids, haystack_sessions)):
        if sid in ans_sids:
            print(f"\n--- Answer Session {sid} ({len(sess)} msgs) ---")
            for m in sess:
                print(f"  {m['role']}: {m['content']}")
