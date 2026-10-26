import json

with open("datasets/external/longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = json.load(f)

it = next(x for x in items if x["question_id"] == "7527f7e2")
for s, sid in zip(it["haystack_sessions"], it["haystack_session_ids"]):
    for idx, turn in enumerate(s):
        if "800" in turn["content"] or "handbag" in turn["content"].lower():
            print(f"FOUND in session {sid}, turn {idx} ({turn['role']}):")
            print(turn["content"])
