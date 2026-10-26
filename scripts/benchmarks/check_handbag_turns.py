import json

with open("datasets/external/longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = json.load(f)

it = next(x for x in items if x["question_id"] == "7527f7e2")
for s, sid in zip(it["haystack_sessions"], it["haystack_session_ids"]):
    if sid == "answer_7cb94507":
        print("--- SESSION ---")
        for turn in s:
            print(f"[{turn['role']}]: {turn['content'][:120]}")
