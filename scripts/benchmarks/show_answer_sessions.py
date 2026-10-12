import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))

for qid in ["gpt4_68e94287", "gpt4_88806d6e", "gpt4_93f6379c"]:
    it = next(item for item in data if item["question_id"] == qid)
    print("=" * 60)
    print(qid, it["question"])
    print("GT:", it["answer"])
    for s_idx, session in enumerate(it["haystack_sessions"]):
        sid = it["haystack_session_ids"][s_idx]
        s_date = it["haystack_dates"][s_idx]
        if "answer" in sid:
            print(f"*** SESSION {sid} ({s_date}) ***")
            for turn in session:
                role = turn.get("role")
                txt = turn.get("content", "")
                print(f"  [{role}]: {txt}")
