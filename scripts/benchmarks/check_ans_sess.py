import json

with open("datasets/external/longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = json.load(f)

it = items[0]
for s, sid in zip(it["haystack_sessions"], it["haystack_session_ids"]):
    if sid == "answer_280352e9":
        print(f"--- SESSION {sid} ---")
        for turn in s:
            print(f"[{turn['role']}]: {turn['content']}")
