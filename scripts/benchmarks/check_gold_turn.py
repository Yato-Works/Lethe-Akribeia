import json
from artificial_memory_gen.compiler.execution_planner import CognitiveExecutionPlanner
from artificial_memory.research.benchmarks.external.longmemeval_adapter import LongMemEvalAdapter

with open("datasets/external/longmemeval_s_cleaned.json", encoding="utf-8") as f:
    items = json.load(f)

it = next(x for x in items if x["question_id"] == "51a45a95")
print("Question:", it["question"])
print("GT:", it["answer"])
print("Answer sessions:", it["answer_session_ids"])

# Print turns in answer sessions
dates = it["haystack_dates"]
sids = it["haystack_session_ids"]
for s_idx, session in enumerate(it["haystack_sessions"]):
    sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
    if sid in it["answer_session_ids"]:
        print(f"--- GOLD SESSION {sid} ({dates[s_idx]}) ---")
        for turn in session:
            print(f"[{turn['role']}]: {turn['content']}")
