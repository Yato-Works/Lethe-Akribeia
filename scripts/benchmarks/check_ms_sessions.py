import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO / "datasets" / "external" / "longmemeval_s_cleaned.json"

with open(DATASET_PATH, encoding="utf-8") as f:
    data = json.load(f)

check_qids = ['36b9f61e', 'edced276', 'f35224e0', '8979f9ec']

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
            for turn in sess:
                speaker = turn.get("role", "user")
                c = turn.get("content", "")
                if speaker == "user":
                    # Check for gold session keywords
                    if qid == '36b9f61e' and any(k in c.lower() for k in ["gucci", "gown", "boots", "$1200", "$800", "$500"]):
                        print(f"  [{sid} on {s_date}] USER: {c.strip()[:200]}")
                    elif qid == 'edced276' and any(k in c.lower() for k in ["hawaii", "new york", "nyc", "days"]):
                        print(f"  [{sid} on {s_date}] USER: {c.strip()[:200]}")
                    elif qid == 'f35224e0' and any(k in c.lower() for k in ["built this", "favorite murder", "episode"]):
                        print(f"  [{sid} on {s_date}] USER: {c.strip()[:200]}")
                    elif qid == '8979f9ec' and any(k in c.lower() for k in ["fajita", "lentil", "lunch"]):
                        print(f"  [{sid} on {s_date}] USER: {c.strip()[:200]}")
