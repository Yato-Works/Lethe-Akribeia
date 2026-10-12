import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
data = json.load(open(REPO / "datasets" / "external" / "longmemeval_s_cleaned.json", encoding="utf-8"))
it = next(item for item in data if item["question_id"] == "gpt4_88806d6e")

import re
for s_idx, session in enumerate(it["haystack_sessions"]):
    sid = it["haystack_session_ids"][s_idx]
    s_date = it["haystack_dates"][s_idx]
    for turn in session:
        if re.search(r"\bTom\b", turn["content"]):
            print(f"[{sid} on {s_date}] {turn['role']}: {turn['content']}")
