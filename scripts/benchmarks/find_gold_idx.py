import json
import re

with open("datasets/external/longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = json.load(f)

it = next(x for x in items if x["question_id"] == "51a45a95")

all_turns = []
for s_idx, session in enumerate(it["haystack_sessions"]):
    sid = it["haystack_session_ids"][s_idx]
    s_date = it["haystack_dates"][s_idx]
    for turn in session:
        all_turns.append((sid, s_date, turn["role"], turn["content"]))

gold_content = "I actually redeemed a $5 coupon on coffee creamer"
for idx, t in enumerate(all_turns):
    if gold_content in t[3]:
        print(f"FOUND GOLD TURN at idx={idx}: sid={t[0]}, role={t[2]}")
