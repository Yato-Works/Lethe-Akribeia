import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))

targets = ["gpt4_1e4a8aec", "gpt4_4929293b", "eac54add", "6e984302"]

for qid in targets:
    it = next(item for item in data if item["question_id"] == qid)
    print("=" * 70)
    print(f"[{qid}] Q_Date: {it.get('question_date')} | Q: {it['question']}")
    print(f"GT: {it['answer']}")
    for s_idx, session in enumerate(it["haystack_sessions"]):
        sid = it["haystack_session_ids"][s_idx]
        s_date = it["haystack_dates"][s_idx]
        if "answer" in sid:
            for turn in session:
                print(f"  [{sid} on {s_date}] {turn['role']}: {turn['content'][:120]}...")
