"""Dump full text of the two gold user turns for 6cb6f249 (UTF-8 safe)."""
import json

with open("datasets/external/longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = json.load(f)
it = next(x for x in items if x["question_id"] == "6cb6f249")
out = []
for sid, sess in zip(it["haystack_session_ids"], it["haystack_sessions"]):
    if sid in ("answer_a4204937_1", "answer_a4204937_2"):
        for t in sess[:2]:
            out.append(f"=== {sid} [{t['role']}] ===\n{t['content']}\n")
with open("benchmark_results/pipeline_v4/gold_6cb6f249_full.txt", "w",
          encoding="utf-8") as f:
    f.write("\n".join(out))
print("\n".join(o[:300] for o in out))
