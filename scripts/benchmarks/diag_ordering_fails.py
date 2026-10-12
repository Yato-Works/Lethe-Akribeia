import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))

targets = ["gpt4_68e94287", "gpt4_88806d6e", "gpt4_93f6379c"]

for qid in targets:
    it = next(item for item in data if item["question_id"] == qid)
    print("=" * 70)
    print(f"[{qid}] {it['question']}")
    print(f"GT: {it['answer']}")
    print("-" * 70)
    for s_idx, session in enumerate(it["haystack_sessions"]):
        s_date = it["haystack_dates"][s_idx]
        sid = it["haystack_session_ids"][s_idx]
        for turn in session:
            c = turn["content"]
            if any(k.lower() in c.lower() for k in ["plank", "chili", "mark and sarah", "meet tom", "met tom", "page turner", "marketing professional"]):
                print(f"[{sid} on {s_date}] {turn['role']}: {c[:140]}...")
