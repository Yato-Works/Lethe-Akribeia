import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    data = json.load(f)

check_qids = ['36b9f61e', 'edced276', 'f35224e0', '8979f9ec', 'a3332713']

for it in data:
    qid = it['question_id']
    if qid in check_qids:
        print("=" * 80)
        print(f"QID: {qid} | Answer: {it['answer']}")
        print(f"Question: {it['question']}")
        sids = it.get("haystack_session_ids", [])
        dates = it.get("haystack_dates", [])
        for s_idx, sess in enumerate(it.get("haystack_sessions", [])):
            sid = sids[s_idx] if s_idx < len(sids) else f"S{s_idx+1}"
            s_date = dates[s_idx] if s_idx < len(dates) else ""
            if sid.startswith("answer_"):
                for turn in sess:
                    speaker = turn.get("role", "user")
                    c = turn.get("content", "")
                    # print sentences mentioning numbers/amounts
                    for sent in re.split(r"(?<=[.!?])\s+", c):
                        if any(k in sent.lower() for k in ["$", "episode", "meal", "lunch", "day", "days"]):
                            if any(k in sent.lower() for k in ["gucci", "gown", "boot", "hawaii", "york", "built this", "murder", "fajita", "lentil", "coworker", "brother"]):
                                print(f"  [{sid}] ({speaker}): {sent.strip()}")
